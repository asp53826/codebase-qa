"""The HTTP API.

Every request goes through the same middleware chain, in this order and for a
reason:

  request id   assigned first so *everything* after it, including the failures,
               is attributable to one request
  auth         before rate limiting, so an unauthenticated flood is rejected
               without consuming anybody's quota
  rate limit   before the budget check, since it's the cheaper of the two
  budget       last, because it needs a database read

Errors return a stable JSON shape with the request id in it, so a user can
paste one line into a support ticket and it's enough to find every log line for
that request.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import AuthError, AuthService
from .db import Database
from .limits import CostTracker, TokenBucketLimiter, cost_micros
from .queue import JobQueue, Worker
from .rag import RagService
from .tracing import Tracer, new_request_id, request_id_var


class AskRequest(BaseModel):
    repo_id: str
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    files: dict[str, str] = Field(description="path -> source text")


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    monthly_cent_budget: int = Field(default=500, ge=0)


class AppState:
    def __init__(self, db_path: str = ":memory:", start_worker: bool = True,
                 rate_capacity: int = 60, rate_refill: float = 1.0,
                 tracer: Tracer | None = None):
        self.db = Database(db_path)
        self.tracer = tracer or Tracer()
        self.auth = AuthService(self.db)
        self.limiter = TokenBucketLimiter(rate_capacity, rate_refill)
        self.costs = CostTracker(self.db)
        self.queue = JobQueue(self.db)
        self.rag = RagService(self.db, self.tracer)
        self.queue.register("index_repo", self._handle_index)
        self.worker = Worker(self.queue) if start_worker else None
        if self.worker:
            self.worker.start()

    def _handle_index(self, payload: dict) -> None:
        """Idempotent on purpose: the queue is at-least-once, so this may run
        twice for one enqueue. Indexing deletes then re-inserts, so a second
        run produces the same state rather than duplicate documents."""
        self.rag.index_repo(payload["repo_id"], payload["files"])

    def shutdown(self) -> None:
        if self.worker:
            self.worker.stop()
        self.db.close()


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = state or AppState(os.environ.get("CQA_DB", ":memory:"))
        yield
        app.state.ctx.shutdown()

    app = FastAPI(title="codebase-qa", version="0.1.0", lifespan=lifespan)

    def ctx(request: Request) -> AppState:
        return request.app.state.ctx

    @app.middleware("http")
    async def observability(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        token = request_id_var.set(rid)
        c: AppState = request.app.state.ctx
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            c.tracer.emit("request_failed", path=request.url.path,
                          error=f"{type(e).__name__}: {e}",
                          duration_ms=round((time.perf_counter() - t0) * 1000, 2))
            raise
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        c.tracer.emit("request", method=request.method, path=request.url.path,
                      status=response.status_code,
                      duration_ms=round((time.perf_counter() - t0) * 1000, 2))
        return response

    @app.exception_handler(AuthError)
    async def auth_error(request: Request, exc: AuthError):
        return JSONResponse(status_code=exc.status,
                            content={"error": exc.message,
                                     "request_id": request_id_var.get()})

    # ---------------------------------------------------------------- gates

    def gate(scope: str, cost: float = 1.0):
        """auth -> rate limit -> budget, returning the key on success."""
        def dep(request: Request, authorization: str = Header(default="")):
            c: AppState = request.app.state.ctx
            raw = authorization.removeprefix("Bearer ").strip()
            key = c.auth.require(raw, scope)

            decision = c.limiter.check(key.key_id, cost)
            if not decision.allowed:
                raise RateLimited(decision)

            status = c.costs.status(key.key_id, key.monthly_cent_budget)
            if status.exceeded:
                raise BudgetExceeded(status)
            return key
        return dep

    class RateLimited(Exception):
        def __init__(self, decision):
            self.decision = decision

    class BudgetExceeded(Exception):
        def __init__(self, status):
            self.status = status

    @app.exception_handler(RateLimited)
    async def rate_limited(request: Request, exc: RateLimited):
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded",
                     "retry_after_seconds": round(exc.decision.retry_after, 2),
                     "request_id": request_id_var.get()},
            headers=exc.decision.headers())

    @app.exception_handler(BudgetExceeded)
    async def budget_exceeded(request: Request, exc: BudgetExceeded):
        return JSONResponse(
            status_code=402,
            content={"error": "monthly budget exhausted",
                     "usage": exc.status.as_dict(),
                     "request_id": request_id_var.get()})

    # ---------------------------------------------------------------- routes

    @app.get("/health")
    async def health(request: Request):
        c = ctx(request)
        return {"status": "ok", "schema_version": c.db.version,
                "queue": c.queue.stats()}

    @app.post("/v1/keys", status_code=201)
    async def create_key(body: CreateKeyRequest, request: Request,
                         authorization: str = Header(default="")):
        c = ctx(request)
        # bootstrap: the first key can be created without one, after that
        # minting keys requires admin
        existing = c.db.one("select count(*) as n from api_keys")["n"]
        if existing:
            c.auth.require(authorization.removeprefix("Bearer ").strip(), "admin")
        try:
            full, key = c.auth.create_key(body.name, set(body.scopes),
                                          body.monthly_cent_budget)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        c.tracer.emit("key_created", key_id=key.key_id, scopes=sorted(key.scopes))
        return {"api_key": full, "key_id": key.key_id,
                "scopes": sorted(key.scopes),
                "warning": "this is the only time the key is shown"}

    @app.delete("/v1/keys/{key_id}")
    async def revoke_key(key_id: str, request: Request,
                         key=Depends(gate("admin"))):
        return {"revoked": ctx(request).auth.revoke(key_id), "key_id": key_id}

    @app.post("/v1/repos", status_code=202)
    async def index_repo(body: IndexRequest, request: Request,
                         key=Depends(gate("write")),
                         idempotency_key: str = Header(default="")):
        """Returns 202 and a job id. Indexing is queued, not done inline."""
        import uuid

        c = ctx(request)
        if not body.files:
            return JSONResponse(status_code=400, content={"error": "no files given"})

        repo_id = uuid.uuid4().hex[:12]
        c.db.execute(
            "insert into repositories (repo_id, owner_key, name, source, status, "
            "created_at) values (?, ?, ?, 'upload', 'pending', ?)",
            (repo_id, key.key_id, body.name, time.time()))
        job_id = c.queue.enqueue("index_repo",
                                 {"repo_id": repo_id, "files": body.files},
                                 idempotency_key=idempotency_key or None)
        c.tracer.emit("repo_queued", repo_id=repo_id, job_id=job_id,
                      files=len(body.files))
        return {"repo_id": repo_id, "job_id": job_id, "status": "pending"}

    @app.get("/v1/repos/{repo_id}")
    async def repo_status(repo_id: str, request: Request, key=Depends(gate("read"))):
        row = ctx(request).db.one("select * from repositories where repo_id = ?",
                                  (repo_id,))
        if row is None:
            return JSONResponse(status_code=404, content={"error": "no such repo"})
        return {"repo_id": row["repo_id"], "name": row["name"],
                "status": row["status"], "doc_count": row["doc_count"],
                "indexed_at": row["indexed_at"], "error": row["error"]}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str, request: Request, key=Depends(gate("read"))):
        s = ctx(request).queue.status(job_id)
        return s or JSONResponse(status_code=404, content={"error": "no such job"})

    @app.post("/v1/ask")
    async def ask(body: AskRequest, request: Request, key=Depends(gate("read", cost=2.0))):
        c = ctx(request)
        rid = request_id_var.get()

        repo = c.db.one("select status from repositories where repo_id = ?",
                        (body.repo_id,))
        if repo is None:
            return JSONResponse(status_code=404, content={"error": "no such repo"})
        if repo["status"] != "ready":
            return JSONResponse(status_code=409,
                                content={"error": f"repo is {repo['status']}, not ready"})

        cached = c.rag.lookup_cache(body.repo_id, body.question)
        if cached is not None:
            c.costs.record(key.key_id, rid, "/v1/ask", c.rag.model, cached=True)
            c.tracer.emit("cache_hit", repo_id=body.repo_id)
            return {"answer": cached.text, "citations": cached.citations,
                    "cached": True, "cost_usd": 0.0, "request_id": rid}

        answer = c.rag.answer(body.repo_id, body.question, body.k)
        micros = c.costs.record(key.key_id, rid, "/v1/ask", c.rag.model,
                                answer.prompt_tokens, answer.completion_tokens)
        c.rag.store_cache(body.repo_id, body.question, answer, micros)
        return {"answer": answer.text, "citations": answer.citations,
                "cached": False, "cost_usd": round(micros / 1e6, 6),
                "request_id": rid}

    @app.get("/v1/usage")
    async def usage(request: Request, key=Depends(gate("read"))):
        c = ctx(request)
        return {"key_id": key.key_id,
                **c.costs.status(key.key_id, key.monthly_cent_budget).as_dict(),
                "by_route": c.costs.breakdown(key.key_id)}

    @app.get("/v1/trace/{request_id}")
    async def trace(request_id: str, request: Request, key=Depends(gate("read"))):
        return {"request_id": request_id, "spans": ctx(request).tracer.trace(request_id)}

    @app.get("/metrics")
    async def metrics(request: Request, key=Depends(gate("admin"))):
        c = ctx(request)
        cache = c.db.one("select count(*) as n, coalesce(sum(hits),0) as hits "
                         "from answer_cache")
        return {
            "queue": c.queue.stats(),
            "dead_letters": len(c.queue.dead_letters()),
            "rate_limiter_keys": c.limiter.tracked_keys,
            "cache": {"entries": cache["n"], "hits": cache["hits"]},
            "timings": c.tracer.timing_summary(),
            "schema_version": c.db.version,
        }

    @app.get("/v1/dead-letters")
    async def dead_letters(request: Request, key=Depends(gate("admin"))):
        return {"jobs": ctx(request).queue.dead_letters()}

    return app


app = create_app()
