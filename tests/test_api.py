"""End-to-end API tests.

These exercise the middleware chain in order — request id, auth, rate limit,
budget — because the order is load bearing and a unit test of each piece in
isolation wouldn't catch getting it wrong.
"""

import pytest
from fastapi.testclient import TestClient

from app.server import AppState, create_app

SAMPLE = {
    "src/parser.py": (
        "import re\n\n"
        "def parse_config(text):\n"
        "    \"\"\"Parse a key=value config block into a dict.\"\"\"\n"
        "    out = {}\n"
        "    for line in text.splitlines():\n"
        "        if '=' in line:\n"
        "            k, v = line.split('=', 1)\n"
        "            out[k.strip()] = v.strip()\n"
        "    return out\n\n"
        "class ConfigError(Exception):\n"
        "    pass\n"
    ),
    "src/cache.py": (
        "import time\n\n"
        "class TTLCache:\n"
        "    def __init__(self, ttl_seconds=60):\n"
        "        self.ttl = ttl_seconds\n"
        "        self._data = {}\n\n"
        "    def get(self, key):\n"
        "        hit = self._data.get(key)\n"
        "        if hit and time.time() - hit[1] < self.ttl:\n"
        "            return hit[0]\n"
        "        return None\n"
    ),
}


@pytest.fixture
def client():
    # worker off so the tests drive the queue explicitly and never race it
    state = AppState(":memory:", start_worker=False)
    with TestClient(create_app(state)) as c:
        c.state = state
        yield c
    state.shutdown()


def admin_key(client) -> str:
    r = client.post("/v1/keys", json={"name": "root", "scopes": ["admin"]})
    assert r.status_code == 201
    return r.json()["api_key"]


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def make_repo(client, key: str) -> str:
    r = client.post("/v1/repos", json={"name": "demo", "files": SAMPLE}, headers=auth(key))
    assert r.status_code == 202
    client.state.queue.drain()
    return r.json()["repo_id"]


# ---------------------------------------------------------------- basics

def test_health_needs_no_key(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["schema_version"] >= 1


def test_every_response_carries_a_request_id(client):
    r = client.get("/health")
    assert r.headers["X-Request-ID"]


def test_supplied_request_id_is_honoured(client):
    """So a caller can correlate their logs with the service's."""
    r = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert r.headers["X-Request-ID"] == "my-trace-123"


# ---------------------------------------------------------------- auth

def test_first_key_bootstraps_then_admin_is_required(client):
    key = admin_key(client)
    assert client.post("/v1/keys", json={"name": "second"}).status_code == 401
    r = client.post("/v1/keys", json={"name": "second"}, headers=auth(key))
    assert r.status_code == 201


def test_key_is_shown_once_with_a_warning(client):
    body = client.post("/v1/keys", json={"name": "x"}).json()
    assert body["api_key"].startswith("cqa_") and "only time" in body["warning"]


def test_no_key_is_401(client):
    admin_key(client)
    assert client.post("/v1/ask", json={"repo_id": "x", "question": "y"}).status_code == 401


def test_bad_key_is_401(client):
    admin_key(client)
    r = client.post("/v1/ask", json={"repo_id": "x", "question": "y"},
                    headers=auth("cqa_deadbeefdeadbeef_nonsense"))
    assert r.status_code == 401 and "request_id" in r.json()


def test_read_key_cannot_index(client):
    root = admin_key(client)
    reader = client.post("/v1/keys", json={"name": "r", "scopes": ["read"]},
                         headers=auth(root)).json()["api_key"]
    r = client.post("/v1/repos", json={"name": "d", "files": SAMPLE}, headers=auth(reader))
    assert r.status_code == 403


def test_revoked_key_stops_working(client):
    root = admin_key(client)
    body = client.post("/v1/keys", json={"name": "temp"}, headers=auth(root)).json()
    assert client.get("/v1/usage", headers=auth(body["api_key"])).status_code == 200
    client.delete(f"/v1/keys/{body['key_id']}", headers=auth(root))
    assert client.get("/v1/usage", headers=auth(body["api_key"])).status_code == 401


# ---------------------------------------------------------------- indexing

def test_indexing_is_queued_not_inline(client):
    """202 with a job id, so a slow repo never holds an HTTP connection open."""
    key = admin_key(client)
    r = client.post("/v1/repos", json={"name": "demo", "files": SAMPLE}, headers=auth(key))
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    assert client.get(f"/v1/repos/{body['repo_id']}",
                      headers=auth(key)).json()["status"] == "pending"

    client.state.queue.drain()
    after = client.get(f"/v1/repos/{body['repo_id']}", headers=auth(key)).json()
    assert after["status"] == "ready" and after["doc_count"] > 0


def test_job_status_is_visible(client):
    key = admin_key(client)
    job_id = client.post("/v1/repos", json={"name": "d", "files": SAMPLE},
                         headers=auth(key)).json()["job_id"]
    assert client.get(f"/v1/jobs/{job_id}", headers=auth(key)).json()["status"] == "queued"
    client.state.queue.drain()
    assert client.get(f"/v1/jobs/{job_id}", headers=auth(key)).json()["status"] == "done"


def test_idempotency_key_does_not_queue_twice(client):
    key = admin_key(client)
    h = {**auth(key), "Idempotency-Key": "repo-abc"}
    a = client.post("/v1/repos", json={"name": "d", "files": SAMPLE}, headers=h).json()
    b = client.post("/v1/repos", json={"name": "d", "files": SAMPLE}, headers=h).json()
    assert a["job_id"] == b["job_id"]


def test_empty_upload_is_rejected(client):
    key = admin_key(client)
    assert client.post("/v1/repos", json={"name": "d", "files": {}},
                       headers=auth(key)).status_code == 400


def test_asking_before_ready_is_409(client):
    key = admin_key(client)
    repo = client.post("/v1/repos", json={"name": "d", "files": SAMPLE},
                       headers=auth(key)).json()["repo_id"]
    r = client.post("/v1/ask", json={"repo_id": repo, "question": "parse"},
                    headers=auth(key))
    assert r.status_code == 409


def test_unknown_repo_is_404(client):
    key = admin_key(client)
    assert client.post("/v1/ask", json={"repo_id": "nope", "question": "x"},
                       headers=auth(key)).status_code == 404


# ---------------------------------------------------------------- asking

def test_ask_returns_an_answer_with_citations(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    body = client.post("/v1/ask", json={"repo_id": repo, "question": "how is config parsed"},
                       headers=auth(key)).json()
    assert body["citations"]
    assert any("parser.py" in c["path"] for c in body["citations"])
    assert all(c["start_line"] <= c["end_line"] for c in body["citations"])


def test_camel_and_snake_case_are_searchable(client):
    """A search for 'ttl cache' must find TTLCache."""
    key = admin_key(client)
    repo = make_repo(client, key)
    body = client.post("/v1/ask", json={"repo_id": repo, "question": "ttl cache expiry"},
                       headers=auth(key)).json()
    assert any("cache.py" in c["path"] for c in body["citations"])


def test_second_identical_question_is_cached_and_free(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    q = {"repo_id": repo, "question": "how is config parsed"}
    first = client.post("/v1/ask", json=q, headers=auth(key)).json()
    second = client.post("/v1/ask", json=q, headers=auth(key)).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["cost_usd"] == 0.0
    assert second["answer"] == first["answer"]


def test_reindexing_invalidates_the_cache(client):
    """The cache key includes the index version, so a re-index can't keep
    serving answers built from the old code."""
    key = admin_key(client)
    repo = make_repo(client, key)
    q = {"repo_id": repo, "question": "how is config parsed"}
    client.post("/v1/ask", json=q, headers=auth(key))
    assert client.post("/v1/ask", json=q, headers=auth(key)).json()["cached"] is True

    client.state.rag.index_repo(repo, {**SAMPLE, "src/new.py": "def added():\n    pass\n"})
    assert client.post("/v1/ask", json=q, headers=auth(key)).json()["cached"] is False


def test_question_length_is_validated(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    assert client.post("/v1/ask", json={"repo_id": repo, "question": ""},
                       headers=auth(key)).status_code == 422
    assert client.post("/v1/ask", json={"repo_id": repo, "question": "x" * 3000},
                       headers=auth(key)).status_code == 422


# ---------------------------------------------------------------- limits

def test_rate_limit_returns_429_with_retry_after(client):
    state = AppState(":memory:", start_worker=False, rate_capacity=3, rate_refill=0.1)
    with TestClient(create_app(state)) as c:
        key = admin_key(c)
        codes = [c.get("/v1/usage", headers=auth(key)).status_code for _ in range(8)]
        assert 429 in codes
        r = c.get("/v1/usage", headers=auth(key))
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) >= 1
        assert r.json()["retry_after_seconds"] > 0
        assert r.headers["X-RateLimit-Remaining"] == "0"
    state.shutdown()


def test_rate_limits_are_per_key(client):
    state = AppState(":memory:", start_worker=False, rate_capacity=2, rate_refill=0.01)
    with TestClient(create_app(state)) as c:
        root = admin_key(c)
        other = c.post("/v1/keys", json={"name": "b", "scopes": ["read"]},
                       headers=auth(root)).json()["api_key"]
        for _ in range(5):
            c.get("/v1/usage", headers=auth(root))
        assert c.get("/v1/usage", headers=auth(root)).status_code == 429
        assert c.get("/v1/usage", headers=auth(other)).status_code == 200
    state.shutdown()


def test_budget_exhaustion_returns_402(client):
    root = admin_key(client)
    broke = client.post("/v1/keys", json={"name": "broke", "scopes": ["read"],
                                          "monthly_cent_budget": 0},
                        headers=auth(root)).json()["api_key"]
    r = client.get("/v1/usage", headers=auth(broke))
    assert r.status_code == 402
    assert r.json()["usage"]["exceeded"] is True


def test_usage_reports_spend_by_route(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    client.post("/v1/ask", json={"repo_id": repo, "question": "parse config"},
                headers=auth(key))
    body = client.get("/v1/usage", headers=auth(key)).json()
    assert body["requests"] >= 1
    assert any(r["route"] == "/v1/ask" for r in body["by_route"])


# ---------------------------------------------------------------- ops

def test_trace_is_retrievable_by_request_id(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    r = client.post("/v1/ask", json={"repo_id": repo, "question": "config"},
                    headers=auth(key))
    rid = r.json()["request_id"]
    spans = client.get(f"/v1/trace/{rid}", headers=auth(key)).json()["spans"]
    assert {s["span"] for s in spans} >= {"answer", "retrieve"}
    assert all(s["duration_ms"] >= 0 for s in spans)


def test_metrics_require_admin(client):
    root = admin_key(client)
    reader = client.post("/v1/keys", json={"name": "r", "scopes": ["read"]},
                         headers=auth(root)).json()["api_key"]
    assert client.get("/metrics", headers=auth(reader)).status_code == 403
    assert client.get("/metrics", headers=auth(root)).status_code == 200


def test_metrics_report_queue_cache_and_timings(client):
    key = admin_key(client)
    repo = make_repo(client, key)
    q = {"repo_id": repo, "question": "config"}
    client.post("/v1/ask", json=q, headers=auth(key))
    client.post("/v1/ask", json=q, headers=auth(key))
    m = client.get("/metrics", headers=auth(key)).json()
    assert m["queue"]["done"] >= 1
    assert m["cache"]["entries"] >= 1 and m["cache"]["hits"] >= 1
    assert "retrieve" in m["timings"]


def test_dead_letters_are_listed(client):
    key = admin_key(client)
    client.state.queue.register("explode",
                                lambda p: (_ for _ in ()).throw(RuntimeError("nope")))
    client.state.queue.enqueue("explode", {}, max_attempts=1)
    client.state.queue.drain()
    body = client.get("/v1/dead-letters", headers=auth(key)).json()
    assert len(body["jobs"]) == 1 and "nope" in body["jobs"][0]["last_error"]
