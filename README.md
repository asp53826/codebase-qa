# codebase-qa

Ask questions about a codebase and get back cited line ranges. The Q&A is the
excuse; the point is everything around it — auth, rate limiting, a durable job
queue, caching, per-request cost tracking, and tracing.

The gap between a prototype and a service isn't the model call. It's that the
prototype has no auth, does its slow work inside the request, retries nothing,
caches nothing, can't tell you what a request cost, and logs `print(e)`.

```bash
pip install -r requirements-dev.txt
pytest                                  # 78 tests
uvicorn app.server:app --port 8000
```

```bash
curl -X POST localhost:8000/v1/keys -d '{"name":"me","scopes":["admin"]}'
curl -X POST localhost:8000/v1/repos -H "authorization: Bearer $KEY" \
     -d '{"name":"demo","files":{"src/a.py":"def parse(x): ..."}}'   # 202 + job_id
curl -X POST localhost:8000/v1/ask -H "authorization: Bearer $KEY" \
     -d '{"repo_id":"...","question":"how is config parsed"}'
```

## The boring parts

### Auth

Keys are stored **hashed with scrypt**, never in plaintext. A database dump
should not be a list of working credentials, so the key is shown exactly once
at creation and is unrecoverable after that.

Three details that are easy to miss:

- **Lookup by public prefix, compare in constant time.** A key is
  `cqa_<public_id>_<secret>`. Hashing every row per request doesn't scale, so
  the id is the index; the secret half is compared with `compare_digest`,
  because `==` on a hash short-circuits and leaks a timing signal.
- **An unknown key id and a wrong secret return the same message**, after doing
  the same work. Otherwise "invalid key" vs "no such key" is an enumeration
  oracle.
- **Revocation is a timestamp, not a `DELETE`.** "When did this key stop
  working" is a question you get asked during an incident.

Scopes are `read`/`write`/`admin`, and `rotate()` issues a replacement carrying
the same name and budget before revoking the old one.

### Rate limiting

Token bucket, not a fixed window. A fixed window lets a caller spend the whole
allowance at 11:59:59 and the whole next one at 12:00:00 — double the limit
across two seconds. A bucket bounds the burst by its capacity and the sustained
rate by its refill.

429s carry a **computed** `Retry-After`, not a guess:

```python
wait = (cost - b.tokens) / self.refill   # exactly when enough tokens exist
```

A 429 without a usable `Retry-After` just makes the client's retry loop tighter.
There's a test that sleeping for exactly the advertised duration succeeds.

Expensive routes cost more tokens (`/v1/ask` costs 2), and idle buckets are
swept so a long tail of one-off keys can't grow memory without bound.

### The job queue

Indexing takes minutes. Doing it in the request means the client times out, a
deploy mid-request loses the work, and one big repo blocks a worker. So
`POST /v1/repos` returns **202 with a job id**.

"A queue" is easy. The properties that make it survivable:

| property | why |
|---|---|
| **durable** — jobs live in SQLite | a restart loses nothing; a thread pool can't promise that |
| **visibility timeout** | a worker killed mid-job doesn't strand it — the lock expires and it's claimable again |
| **at-least-once, not at-most-once** | a crash after the work but before the commit reruns the job. That's the correct trade here, and it's why handlers must be idempotent |
| **exponential backoff with full jitter** | fixed retries against a downed dependency produce a thundering herd that keeps it down |
| **dead letter queue** | after `max_attempts` the job stops and is *kept*. Silently dropping it loses the evidence |
| **idempotency keys** | the client that timed out is the one most likely to retry, and it should get the original job back |

Claiming is a conditional `UPDATE`, so two workers racing can't both win — the
loser's `WHERE` clause no longer matches:

```sql
update jobs set status='running', attempts=attempts+1, locked_until=?
where job_id=? and (status='queued' or (status='running' and locked_until<?))
```

Backoff uses **full jitter** (uniform over `[0, delay]`) rather than
`delay ± a bit`, because the point is decorrelating a fleet of retrying
workers and a narrow band around a common value barely does that.

The index handler deletes then re-inserts, so a duplicate delivery produces the
same state rather than doubled documents.

### Caching

The cache key is `(repo_id, indexed_at, question)` — **the index version is in
the key**. Caching on the question alone means a repo that was just re-indexed
keeps serving answers built from the old code. Re-indexing changes `indexed_at`
and invalidates every entry for that repo implicitly, with no sweep:

```python
def test_reindexing_invalidates_the_cache(client):
    client.post("/v1/ask", json=q, headers=auth(key))
    assert client.post("/v1/ask", json=q, headers=auth(key)).json()["cached"] is True
    client.state.rag.index_repo(repo, {**SAMPLE, "src/new.py": "..."})
    assert client.post("/v1/ask", json=q, headers=auth(key)).json()["cached"] is False
```

A hit still writes a usage row, with `cost_micros=0` and `cached=1`, so the hit
rate stays visible and a cached request never bills the caller.

### Cost tracking

Per-request usage rows, not a running counter, so "why is my bill high" is
answerable per route and per request id rather than being one number nobody can
decompose.

Prices are **micro-dollars per token as integers**. Floats accumulate error over
millions of rows, and a billing number that drifts is worse than one that's
slightly stale. An unknown model falls back to a real price rather than zero —
silently pricing something at zero hides spend.

Budgets are enforced per key over a rolling 30 days, returning **402** with the
usage breakdown when exhausted.

### Observability

Every log line is JSON with a request id. `"Error processing request"` appearing
400 times in a text log tells you nothing; the same line with a request id lets
you pull every event for the one request a user is complaining about.

Secrets are redacted on the way out, so it doesn't depend on the caller
remembering:

```python
SECRET_PATTERNS = [re.compile(r"\bcqa_[a-f0-9]{16}_[A-Za-z0-9_\-]{20,}"), ...]
SECRET_FIELDS = {"authorization", "api_key", "token", "secret", ...}
```

Tests assert a real generated key doesn't survive redaction, including nested
inside dicts and lists.

Spans nest with parent ids, `GET /v1/trace/{request_id}` returns the whole tree
for one request, and `/metrics` reports p50/p95 per span name — which is what
you actually want when asked why the endpoint got slow.

### Middleware order

The chain is load bearing, so it's tested end to end rather than per-piece:

1. **request id** — first, so even the failures are attributable
2. **auth** — before rate limiting, so an unauthenticated flood doesn't consume anyone's quota
3. **rate limit** — before the budget check, since it's the cheaper of the two
4. **budget** — last, because it needs a database read

## Tests

78 tests. The ones worth calling out:

- a plaintext key never appears in the stored hash
- unknown key id and wrong secret produce identical errors
- a revoked key stops working immediately, and the row survives
- idling for an hour doesn't buy an hour of burst
- sleeping for exactly the advertised `Retry-After` succeeds
- rate limits and budgets are per key
- **a claimed job is invisible to a second worker**
- **an expired lock makes a job claimable again** (the killed-worker case)
- a job survives being handed to a completely new queue object
- retries respect backoff, then dead-letter, then can be requeued
- an idempotency key doesn't queue twice
- re-indexing invalidates the cache
- a generated key doesn't survive log redaction, nested or not
- spans nest, record exceptions, and are isolated per request

## Being straight about what this is

- **Not deployed and it has no users.** The original brief said "deploy it and
  get a few real users" and I've done neither — no hosting account, no
  audience. Everything above is tested locally and in CI, and I'd rather say
  that than imply traffic I don't have.
- **SQLite, not Postgres.** Deliberate: this runs from a clone with no
  docker-compose, and WAL, busy timeouts, transactions and conditional updates
  are the same shape you'd write against Postgres. It would not survive
  multi-node writers.
- **The queue worker is a thread in the API process.** Fine for one box, wrong
  for anything real — the queue is durable and claim-safe precisely so workers
  can be separate processes, but I haven't split them out.
- **Retrieval is BM25 over code chunks**, no embeddings and no LLM in the
  default path, so the whole service runs with no API key. Chunking splits at
  `def`/`class` boundaries and tokenization splits camelCase and snake_case, so
  "ttl cache" finds `TTLCache`. Retrieval *quality* is the subject of
  [rag-eval](https://github.com/asp53826/rag-eval), not this repo — an LLM
  generator drops into `RagService.answer` without touching any of the
  infrastructure above, which is the point of keeping them separate.
- **Answers are extractive**: cited code, not prose.
- **No web UI.** API only.
- **In-memory rate limiter**, so limits are per process. Multi-instance needs
  Redis.

## Layout

```
app/db.py        versioned migrations, WAL, transactions
app/auth.py      scrypt-hashed keys, scopes, constant-time compare, rotation
app/limits.py    token bucket, integer-micro pricing, rolling budgets
app/queue.py     durable jobs: visibility timeout, backoff, DLQ, idempotency
app/tracing.py   JSON logs, request ids, nested spans, secret redaction
app/rag.py       code-aware chunking, BM25, version-keyed answer cache
app/server.py    FastAPI, middleware chain, error shapes
tests/           78 tests
```
