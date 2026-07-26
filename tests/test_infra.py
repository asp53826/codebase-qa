"""Tests for the boring parts, which is where the bugs live.

Auth, rate limiting, the job queue, cost accounting and log redaction. None of
this is interesting until it's wrong, at which point it's the only thing that
matters.
"""

import time

import pytest

from app.auth import AuthError, AuthService, generate_key, parse_key
from app.db import Database
from app.limits import CostTracker, TokenBucketLimiter, cost_micros
from app.queue import DEAD, DONE, QUEUED, JobQueue
from app.tracing import Tracer, redact, request_id_var


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


# ---------------------------------------------------------------- migrations

def test_migrations_apply_once(db):
    assert db.version == 1
    assert db.migrate() == 0, "re-running migrations must be a no-op"


def test_transaction_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.tx() as c:
            c.execute("insert into api_keys (key_id, key_hash, name, scopes, "
                      "created_at) values ('x','h','n','read',1)")
            raise RuntimeError("boom")
    assert db.one("select count(*) as n from api_keys")["n"] == 0


# ---------------------------------------------------------------- auth

def test_key_roundtrip(db):
    auth = AuthService(db)
    full, key = auth.create_key("ci", {"read", "write"})
    got = auth.authenticate(full)
    assert got.key_id == key.key_id and got.can("write")


def test_plaintext_key_is_never_stored(db):
    """A database dump must not be a list of working credentials."""
    auth = AuthService(db)
    full, key = auth.create_key("ci")
    _, _, secret = parse_key(full)[0], None, parse_key(full)[1]
    row = db.one("select key_hash from api_keys where key_id = ?", (key.key_id,))
    assert secret not in row["key_hash"]
    assert full not in row["key_hash"]


def test_wrong_secret_is_rejected(db):
    auth = AuthService(db)
    full, key = auth.create_key("ci")
    tampered = full[:-4] + "aaaa"
    with pytest.raises(AuthError):
        auth.authenticate(tampered)


def test_unknown_key_and_bad_secret_give_the_same_message(db):
    """An unknown id must not be distinguishable from a wrong secret."""
    auth = AuthService(db)
    full, _ = auth.create_key("ci")
    with pytest.raises(AuthError) as a:
        auth.authenticate("cqa_0000000000000000_nope")
    with pytest.raises(AuthError) as b:
        auth.authenticate(full[:-4] + "zzzz")
    assert a.value.message == b.value.message


def test_malformed_keys_are_rejected(db):
    auth = AuthService(db)
    for bad in ("", "garbage", "cqa_only_two", "wrong_abc_def", "cqa__secret"):
        with pytest.raises(AuthError):
            auth.authenticate(bad)


def test_revocation_is_immediate_and_recorded(db):
    auth = AuthService(db)
    full, key = auth.create_key("ci")
    assert auth.revoke(key.key_id) is True
    with pytest.raises(AuthError, match="revoked"):
        auth.authenticate(full)
    # kept, not deleted: "when did this stop working" is an incident question
    assert db.one("select revoked_at from api_keys where key_id = ?",
                  (key.key_id,))["revoked_at"] is not None
    assert auth.revoke(key.key_id) is False


def test_scopes_are_enforced(db):
    auth = AuthService(db)
    full, _ = auth.create_key("reader", {"read"})
    auth.require(full, "read")
    with pytest.raises(AuthError) as e:
        auth.require(full, "write")
    assert e.value.status == 403


def test_admin_implies_everything(db):
    auth = AuthService(db)
    full, _ = auth.create_key("root", {"admin"})
    for scope in ("read", "write", "admin"):
        auth.require(full, scope)


def test_rotation_issues_a_new_key_and_kills_the_old(db):
    auth = AuthService(db)
    old, key = auth.create_key("ci", {"write"}, monthly_cent_budget=1234)
    new, new_key = auth.rotate(key.key_id)
    assert new != old
    assert auth.authenticate(new).monthly_cent_budget == 1234
    assert auth.authenticate(new).can("write")
    with pytest.raises(AuthError):
        auth.authenticate(old)


def test_unknown_scope_is_rejected(db):
    with pytest.raises(ValueError, match="unknown scope"):
        AuthService(db).create_key("x", {"superuser"})


def test_keys_are_unique():
    assert len({generate_key()[0] for _ in range(200)}) == 200


# ---------------------------------------------------------------- rate limit

def test_bucket_allows_a_burst_then_throttles():
    clock = [0.0]
    lim = TokenBucketLimiter(5, 1.0, clock=lambda: clock[0])
    assert all(lim.check("k").allowed for _ in range(5))
    assert not lim.check("k").allowed


def test_bucket_refills_over_time():
    clock = [0.0]
    lim = TokenBucketLimiter(2, 1.0, clock=lambda: clock[0])
    lim.check("k"); lim.check("k")
    assert not lim.check("k").allowed
    clock[0] += 1.0
    assert lim.check("k").allowed


def test_refill_is_capped_at_capacity():
    """Idling for an hour must not buy an hour's worth of burst."""
    clock = [0.0]
    lim = TokenBucketLimiter(3, 1.0, clock=lambda: clock[0])
    lim.check("k")
    clock[0] += 10_000
    assert sum(lim.check("k").allowed for _ in range(10)) == 3


def test_keys_are_limited_independently():
    lim = TokenBucketLimiter(2, 1.0)
    lim.check("a"); lim.check("a")
    assert not lim.check("a").allowed
    assert lim.check("b").allowed


def test_retry_after_is_accurate():
    """A 429 without a usable Retry-After just makes the retry loop tighter."""
    clock = [0.0]
    lim = TokenBucketLimiter(1, 0.5, clock=lambda: clock[0])
    lim.check("k")
    d = lim.check("k")
    assert not d.allowed
    assert d.retry_after == pytest.approx(2.0)
    assert d.headers()["Retry-After"] == "2"
    clock[0] += d.retry_after
    assert lim.check("k").allowed


def test_headers_are_present_on_success_too():
    h = TokenBucketLimiter(10, 1.0).check("k").headers()
    assert h["X-RateLimit-Limit"] == "10" and "Retry-After" not in h


def test_cost_weighted_calls():
    lim = TokenBucketLimiter(10, 1.0)
    assert lim.check("k", cost=8).allowed
    assert not lim.check("k", cost=5).allowed
    assert lim.check("k", cost=2).allowed


def test_idle_buckets_are_swept():
    clock = [0.0]
    lim = TokenBucketLimiter(5, 1.0, clock=lambda: clock[0], idle_seconds=100)
    for i in range(50):
        lim.check(f"key{i}")
    assert lim.tracked_keys == 50
    clock[0] += 200
    assert lim.sweep() == 50 and lim.tracked_keys == 0


def test_invalid_limiter_config():
    with pytest.raises(ValueError):
        TokenBucketLimiter(0, 1.0)
    with pytest.raises(ValueError):
        TokenBucketLimiter(5, 0)


# ---------------------------------------------------------------- cost

def test_pricing_is_integer_micros():
    assert cost_micros("gpt-4o-mini", 1000, 500) == 1000 * 15 + 500 * 60
    assert cost_micros("local", 10_000, 10_000) == 0
    assert isinstance(cost_micros("gpt-4o", 3, 4), int)


def test_unknown_model_falls_back_rather_than_charging_zero():
    """Silently pricing an unknown model at zero hides spend."""
    assert cost_micros("some-new-model", 1000, 0) > 0


def test_budget_accumulates_and_trips(db):
    tracker = CostTracker(db)
    for i in range(5):
        tracker.record("k1", f"r{i}", "/v1/ask", "gpt-4o", 1000, 500)
    s = tracker.status("k1", monthly_cent_budget=1)  # $0.01
    assert s.requests == 5 and s.spent_micros == 5 * cost_micros("gpt-4o", 1000, 500)
    assert s.exceeded


def test_cached_requests_are_free_but_still_counted(db):
    tracker = CostTracker(db)
    tracker.record("k1", "r1", "/v1/ask", "gpt-4o", 5000, 5000, cached=True)
    s = tracker.status("k1", 500)
    assert s.spent_micros == 0 and s.requests == 1


def test_budgets_are_per_key(db):
    tracker = CostTracker(db)
    tracker.record("k1", "r1", "/v1/ask", "gpt-4o", 100_000, 0)
    assert tracker.status("k2", 500).spent_micros == 0


def test_spend_outside_the_window_is_excluded(db):
    now = [time.time()]
    tracker = CostTracker(db, clock=lambda: now[0])
    tracker.record("k1", "old", "/v1/ask", "gpt-4o", 100_000, 0)
    now[0] += 31 * 86400
    assert tracker.status("k1", 500).spent_micros == 0


def test_breakdown_is_per_route(db):
    tracker = CostTracker(db)
    tracker.record("k1", "r1", "/v1/ask", "gpt-4o", 1000, 0)
    tracker.record("k1", "r2", "/v1/repos", "gpt-4o", 10, 0)
    routes = {r["route"]: r for r in tracker.breakdown("k1")}
    assert routes["/v1/ask"]["cost_usd"] > routes["/v1/repos"]["cost_usd"]


# ---------------------------------------------------------------- job queue

def test_enqueue_and_run(db):
    q = JobQueue(db)
    seen = []
    q.register("greet", lambda p: seen.append(p["name"]))
    q.enqueue("greet", {"name": "world"})
    job = q.run_one()
    assert job.status == DONE and seen == ["world"]
    assert q.run_one() is None


def test_jobs_survive_a_new_queue_object(db):
    """Durability: the work lives in the database, not in a process."""
    JobQueue(db).enqueue("greet", {"name": "x"})
    q2 = JobQueue(db)
    seen = []
    q2.register("greet", lambda p: seen.append(p["name"]))
    assert q2.run_one().status == DONE and seen == ["x"]


def test_failure_retries_with_backoff(db):
    now = [1000.0]
    q = JobQueue(db, clock=lambda: now[0], base_backoff=10.0)
    q.register("flaky", lambda p: (_ for _ in ()).throw(RuntimeError("nope")))
    jid = q.enqueue("flaky", {}, max_attempts=3)

    job = q.run_one()
    assert job.status == QUEUED and q.status(jid)["attempts"] == 1
    assert q.run_one() is None, "must not retry before its backoff elapses"

    now[0] += 1000
    q.run_one()
    now[0] += 1000
    q.run_one()
    assert q.status(jid)["status"] == DEAD


def test_dead_letters_keep_the_error(db):
    q = JobQueue(db, base_backoff=0.0)
    q.register("bad", lambda p: (_ for _ in ()).throw(ValueError("specific detail")))
    q.enqueue("bad", {"x": 1}, max_attempts=1)
    q.run_one()
    dead = q.dead_letters()
    assert len(dead) == 1 and "specific detail" in dead[0]["last_error"]
    assert dead[0]["payload"] == {"x": 1}


def test_dead_letter_can_be_requeued(db):
    """The usual shape: a job dies because a dependency was down, somebody
    fixes the dependency, and the work should be recoverable rather than lost."""
    attempts = []

    def handler(payload):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("dependency was down")

    q = JobQueue(db, base_backoff=0.0)
    q.register("sometimes", handler)
    jid = q.enqueue("sometimes", {}, max_attempts=1)

    q.run_one()
    assert q.status(jid)["status"] == DEAD

    assert q.requeue_dead(jid) is True
    assert q.run_one().status == DONE
    assert len(attempts) == 2
    assert q.requeue_dead(jid) is False, "only dead jobs can be requeued"


def test_missing_handler_is_not_a_crash(db):
    q = JobQueue(db, base_backoff=0.0)
    jid = q.enqueue("nobody-handles-this", {}, max_attempts=1)
    q.run_one()
    assert q.status(jid)["status"] == DEAD
    assert "no handler" in q.status(jid)["last_error"]


def test_idempotency_key_prevents_duplicates(db):
    q = JobQueue(db)
    a = q.enqueue("index", {"repo": 1}, idempotency_key="repo-1")
    b = q.enqueue("index", {"repo": 1}, idempotency_key="repo-1")
    assert a == b
    assert db.one("select count(*) as n from jobs")["n"] == 1


def test_no_idempotency_key_means_separate_jobs(db):
    q = JobQueue(db)
    assert q.enqueue("index", {}) != q.enqueue("index", {})


def test_delayed_jobs_are_not_claimed_early(db):
    now = [500.0]
    q = JobQueue(db, clock=lambda: now[0])
    q.register("later", lambda p: None)
    q.enqueue("later", {}, delay=100)
    assert q.claim() is None
    now[0] += 101
    assert q.claim() is not None


def test_a_claimed_job_is_invisible_to_other_workers(db):
    now = [0.0]
    a = JobQueue(db, visibility_seconds=60, clock=lambda: now[0], worker_id="a")
    b = JobQueue(db, visibility_seconds=60, clock=lambda: now[0], worker_id="b")
    a.enqueue("work", {})
    assert a.claim() is not None
    assert b.claim() is None, "two workers must not hold the same job"


def test_an_expired_lock_makes_the_job_claimable_again(db):
    """A worker killed mid-job must not strand it forever."""
    now = [0.0]
    a = JobQueue(db, visibility_seconds=30, clock=lambda: now[0], worker_id="a")
    b = JobQueue(db, visibility_seconds=30, clock=lambda: now[0], worker_id="b")
    a.enqueue("work", {})
    a.claim()          # worker a then dies without completing
    now[0] += 31
    reclaimed = b.claim()
    assert reclaimed is not None and reclaimed.attempts == 2


def test_backoff_is_bounded_and_jittered(db):
    q = JobQueue(db, base_backoff=1.0, max_backoff=60.0)
    for attempt in range(1, 12):
        assert 0 <= q.backoff(attempt) <= 60.0
    # jitter: repeated draws at the same attempt must not be identical
    assert len({round(q.backoff(6), 6) for _ in range(20)}) > 1


def test_drain_processes_everything(db):
    q = JobQueue(db)
    done = []
    q.register("t", lambda p: done.append(p["i"]))
    for i in range(25):
        q.enqueue("t", {"i": i})
    assert q.drain() == 25
    assert sorted(done) == list(range(25))
    assert q.stats()["done"] == 25


# ---------------------------------------------------------------- tracing

def test_secrets_are_redacted():
    key = generate_key()[0]
    assert key not in redact(f"failed to auth with {key}")
    assert redact({"authorization": "Bearer abc123xyz789"})["authorization"] == "[redacted]"
    assert redact({"api_key": "anything"})["api_key"] == "[redacted]"
    assert "sk-" not in redact("openai key sk-abcdefghijklmnop1234")


def test_redaction_reaches_nested_structures():
    key = generate_key()[0]
    out = redact({"ctx": {"items": [f"key={key}"]}})
    assert key not in str(out)


def test_redaction_leaves_ordinary_values_alone():
    assert redact({"path": "app/server.py", "count": 5}) == {"path": "app/server.py",
                                                             "count": 5}


def test_spans_nest_and_time(capsys):
    tracer = Tracer(stream=open("/dev/null", "w"))
    request_id_var.set("req-1")
    with tracer.span("outer", repo="x"):
        with tracer.span("inner"):
            time.sleep(0.01)
    spans = {s["span"]: s for s in tracer.trace("req-1")}
    assert spans["inner"]["parent_id"] == spans["outer"]["span_id"]
    assert spans["outer"]["duration_ms"] >= spans["inner"]["duration_ms"]


def test_span_records_an_exception_and_reraises():
    tracer = Tracer(stream=open("/dev/null", "w"))
    request_id_var.set("req-2")
    with pytest.raises(ValueError):
        with tracer.span("failing"):
            raise ValueError("kaboom")
    assert "kaboom" in tracer.trace("req-2")[0]["error"]


def test_traces_are_isolated_by_request():
    tracer = Tracer(stream=open("/dev/null", "w"))
    request_id_var.set("a")
    with tracer.span("one"):
        pass
    request_id_var.set("b")
    with tracer.span("two"):
        pass
    assert [s["span"] for s in tracer.trace("a")] == ["one"]


def test_timing_summary_reports_percentiles():
    tracer = Tracer(stream=open("/dev/null", "w"))
    request_id_var.set("t")
    for _ in range(10):
        with tracer.span("repeated"):
            pass
    s = tracer.timing_summary()["repeated"]
    assert s["count"] == 10 and s["p95_ms"] >= s["p50_ms"]


def test_span_buffer_is_bounded():
    tracer = Tracer(stream=open("/dev/null", "w"), keep=50)
    request_id_var.set("x")
    for _ in range(200):
        with tracer.span("s"):
            pass
    assert len(tracer.spans) == 50
