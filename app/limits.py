"""Rate limiting and cost budgets.

Two different controls that people conflate:

  rate limit   protects the service from one caller's burst
  cost budget  protects the *caller* from their own bill

A caller can be well under the rate limit and still spend $400 in an hour, so
both are enforced.

The rate limiter is a token bucket rather than a fixed window. A fixed window
lets a caller send the whole allowance at 11:59:59 and the whole next one at
12:00:00, which is double the limit across two seconds. A bucket refills
continuously, so the burst is bounded by the bucket size and the sustained rate
by the refill rate.

Both return the information a client needs to behave well: how much is left,
when it resets, and how long to wait. A 429 with no Retry-After just produces a
tighter retry loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class RateDecision:
    allowed: bool
    remaining: float
    limit: int
    reset_after: float
    retry_after: float = 0.0

    def headers(self) -> dict[str, str]:
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
            "X-RateLimit-Reset": str(int(self.reset_after)),
        }
        if not self.allowed:
            h["Retry-After"] = str(max(1, int(self.retry_after + 0.999)))
        return h


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """capacity = burst size, refill_per_second = sustained rate.

    Buckets are created lazily and swept when idle, so a large number of
    one-off keys can't grow memory without bound.
    """

    def __init__(self, capacity: int = 60, refill_per_second: float = 1.0,
                 clock=time.monotonic, idle_seconds: float = 3600.0):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self.capacity = capacity
        self.refill = refill_per_second
        self.clock = clock
        self.idle_seconds = idle_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> RateDecision:
        now = self.clock()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(float(self.capacity), now)
                self._buckets[key] = b
            else:
                elapsed = max(0.0, now - b.updated)
                b.tokens = min(self.capacity, b.tokens + elapsed * self.refill)
                b.updated = now

            full_in = (self.capacity - b.tokens) / self.refill
            if b.tokens >= cost:
                b.tokens -= cost
                return RateDecision(True, b.tokens, self.capacity, full_in)

            # exactly how long until enough tokens exist, not a guess
            wait = (cost - b.tokens) / self.refill
            return RateDecision(False, b.tokens, self.capacity, full_in, retry_after=wait)

    def sweep(self) -> int:
        """Drop buckets that have been idle long enough to be full anyway."""
        now = self.clock()
        with self._lock:
            stale = [k for k, b in self._buckets.items()
                     if now - b.updated > self.idle_seconds]
            for k in stale:
                del self._buckets[k]
            return len(stale)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    @property
    def tracked_keys(self) -> int:
        return len(self._buckets)


# Prices in micro-dollars per token, so everything stays integer arithmetic.
# Floats accumulate error over millions of rows and a billing number that
# drifts is worse than one that's slightly stale.
PRICING = {
    "gpt-4o-mini":            {"prompt": 15, "completion": 60},
    "gpt-4o":                 {"prompt": 250, "completion": 1000},
    "text-embedding-3-small": {"prompt": 2, "completion": 0},
    "local":                  {"prompt": 0, "completion": 0},
}
DEFAULT_MODEL = "gpt-4o-mini"


def cost_micros(model: str, prompt_tokens: int, completion_tokens: int = 0) -> int:
    p = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return prompt_tokens * p["prompt"] + completion_tokens * p["completion"]


@dataclass
class BudgetStatus:
    spent_micros: int
    budget_micros: int
    requests: int

    @property
    def remaining_micros(self) -> int:
        return max(0, self.budget_micros - self.spent_micros)

    @property
    def exceeded(self) -> bool:
        return self.spent_micros >= self.budget_micros

    @property
    def fraction_used(self) -> float:
        return self.spent_micros / self.budget_micros if self.budget_micros else 1.0

    def as_dict(self) -> dict:
        return {
            "spent_usd": round(self.spent_micros / 1e6, 6),
            "budget_usd": round(self.budget_micros / 1e6, 4),
            "remaining_usd": round(self.remaining_micros / 1e6, 6),
            "fraction_used": round(self.fraction_used, 4),
            "requests": self.requests,
            "exceeded": self.exceeded,
        }


class CostTracker:
    """Per-request usage rows, aggregated into a rolling 30 day budget.

    Rows rather than a running counter, so "why is my bill high" is answerable
    per route and per request id instead of being a single number nobody can
    decompose.
    """

    WINDOW_SECONDS = 30 * 86400

    def __init__(self, db, clock=time.time):
        self.db = db
        self.clock = clock

    def record(self, key_id: str, request_id: str, route: str, model: str,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               embed_tokens: int = 0, cached: bool = False) -> int:
        micros = 0 if cached else cost_micros(model, prompt_tokens + embed_tokens,
                                              completion_tokens)
        self.db.execute(
            "insert into usage (key_id, request_id, route, prompt_tokens, "
            "completion_tokens, embed_tokens, cost_micros, cached, created_at) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key_id, request_id, route, prompt_tokens, completion_tokens,
             embed_tokens, micros, int(cached), self.clock()),
        )
        return micros

    def status(self, key_id: str, monthly_cent_budget: int) -> BudgetStatus:
        since = self.clock() - self.WINDOW_SECONDS
        row = self.db.one(
            "select coalesce(sum(cost_micros), 0) as spent, count(*) as n "
            "from usage where key_id = ? and created_at >= ?", (key_id, since))
        return BudgetStatus(int(row["spent"]), monthly_cent_budget * 10_000,
                            int(row["n"]))

    def breakdown(self, key_id: str) -> list[dict]:
        since = self.clock() - self.WINDOW_SECONDS
        return [
            {"route": r["route"], "requests": r["n"],
             "cost_usd": round(r["spent"] / 1e6, 6),
             "cached_requests": r["cached"]}
            for r in self.db.query(
                "select route, count(*) as n, sum(cost_micros) as spent, "
                "sum(cached) as cached from usage "
                "where key_id = ? and created_at >= ? group by route "
                "order by spent desc", (key_id, since))
        ]
