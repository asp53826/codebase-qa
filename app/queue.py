"""A durable background job queue.

Indexing a repository takes minutes. Doing it inside the HTTP request means the
client times out, a deploy mid-request loses the work, and one big repo blocks
a worker thread. So it goes on a queue.

"A queue" is easy; the properties that make it survivable are not:

  durable          jobs live in the database. A process restart loses nothing,
                   which an in-memory list or a thread pool cannot promise.

  visibility timeout
                   claiming a job locks it for a bounded time rather than
                   forever. A worker that is killed mid-job does not strand
                   that job, it becomes claimable again once the lock expires.
                   This is what makes at-least-once delivery work.

  at-least-once, not at-most-once
                   a crash after the work but before the commit means the job
                   runs twice. That is the correct trade for indexing, and it
                   is why handlers must be idempotent. Exactly-once across a
                   process boundary is not available.

  exponential backoff with jitter
                   a failing dependency plus fixed retries produces a
                   thundering herd that keeps it down. Jitter spreads the
                   retries out.

  dead letter      after max_attempts a job stops retrying and is kept for
                   inspection. Silently dropping it loses the evidence.

  idempotency key  submitting the same work twice returns the original job
                   rather than queuing a duplicate, which matters because the
                   client that timed out is the one most likely to retry.

Claiming is a conditional UPDATE, so two workers racing for the same job cannot
both win: SQLite serializes the writes and the loser's WHERE clause no longer
matches.
"""

from __future__ import annotations

import json
import random
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Callable

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
DEAD = "dead"


@dataclass
class Job:
    job_id: str
    kind: str
    payload: dict
    status: str
    attempts: int
    max_attempts: int
    last_error: str | None = None

    @property
    def attempts_left(self) -> int:
        return max(0, self.max_attempts - self.attempts)


class JobQueue:
    def __init__(self, db, visibility_seconds: float = 300.0,
                 base_backoff: float = 2.0, max_backoff: float = 300.0,
                 clock=time.time, worker_id: str | None = None):
        self.db = db
        self.visibility = visibility_seconds
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.clock = clock
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._handlers: dict[str, Callable[[dict], None]] = {}

    def register(self, kind: str, handler: Callable[[dict], None]) -> None:
        self._handlers[kind] = handler

    def enqueue(self, kind: str, payload: dict, max_attempts: int = 3,
                delay: float = 0.0, idempotency_key: str | None = None) -> str:
        now = self.clock()
        if idempotency_key:
            existing = self.db.one(
                "select job_id from jobs where idempotency_key = ?", (idempotency_key,))
            if existing:
                # the client that timed out is the one most likely to retry,
                # and it should get the original job back, not a second one
                return existing["job_id"]

        job_id = uuid.uuid4().hex
        self.db.execute(
            "insert into jobs (job_id, kind, payload, status, attempts, max_attempts, "
            "run_after, created_at, updated_at, idempotency_key) "
            "values (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
            (job_id, kind, json.dumps(payload), QUEUED, max_attempts,
             now + delay, now, now, idempotency_key),
        )
        return job_id

    def claim(self) -> Job | None:
        """Atomically take one ready job. Returns None if there's nothing to do.

        Ready means queued (or a running job whose lock has expired) and past
        its run_after. The UPDATE ... WHERE status/lock still match is what
        makes the claim safe against another worker doing the same thing.
        """
        now = self.clock()
        row = self.db.one(
            "select * from jobs where run_after <= ? and ("
            "  status = ? or (status = ? and locked_until is not null and locked_until < ?)"
            ") order by run_after limit 1",
            (now, QUEUED, RUNNING, now))
        if row is None:
            return None

        cur = self.db.execute(
            "update jobs set status = ?, attempts = attempts + 1, worker = ?, "
            "locked_until = ?, updated_at = ? "
            "where job_id = ? and (status = ? or (status = ? and locked_until < ?))",
            (RUNNING, self.worker_id, now + self.visibility, now,
             row["job_id"], QUEUED, RUNNING, now))
        if cur.rowcount == 0:
            return None  # another worker won the race

        return Job(row["job_id"], row["kind"], json.loads(row["payload"]),
                   RUNNING, row["attempts"] + 1, row["max_attempts"], row["last_error"])

    def complete(self, job_id: str) -> None:
        self.db.execute(
            "update jobs set status = ?, locked_until = null, updated_at = ?, "
            "last_error = null where job_id = ?", (DONE, self.clock(), job_id))

    def fail(self, job_id: str, error: str) -> str:
        """Schedule a retry, or dead-letter if the attempts are used up."""
        row = self.db.one("select attempts, max_attempts from jobs where job_id = ?",
                          (job_id,))
        if row is None:
            return DEAD
        now = self.clock()
        if row["attempts"] >= row["max_attempts"]:
            self.db.execute(
                "update jobs set status = ?, locked_until = null, updated_at = ?, "
                "last_error = ? where job_id = ?", (DEAD, now, error[:4000], job_id))
            return DEAD

        delay = self.backoff(row["attempts"])
        self.db.execute(
            "update jobs set status = ?, locked_until = null, run_after = ?, "
            "updated_at = ?, last_error = ? where job_id = ?",
            (QUEUED, now + delay, now, error[:4000], job_id))
        return QUEUED

    def backoff(self, attempt: int) -> float:
        """Exponential, capped, with full jitter.

        Full jitter (uniform over [0, delay]) rather than delay +/- a bit,
        because the point is to decorrelate a whole fleet of retrying workers,
        and a narrow band around a common value barely does that.
        """
        ceiling = min(self.max_backoff, self.base_backoff * (2 ** max(0, attempt - 1)))
        return random.uniform(0, ceiling)

    def run_one(self) -> Job | None:
        """Claim and execute a single job. Returns it, or None if idle."""
        job = self.claim()
        if job is None:
            return None
        handler = self._handlers.get(job.kind)
        if handler is None:
            self.fail(job.job_id, f"no handler registered for kind {job.kind!r}")
            job.status = FAILED
            return job
        try:
            handler(job.payload)
            self.complete(job.job_id)
            job.status = DONE
        except Exception:
            job.status = self.fail(job.job_id, traceback.format_exc())
        return job

    def drain(self, limit: int = 1000) -> int:
        """Run until there is nothing ready. Used by tests and the CLI worker."""
        done = 0
        while done < limit and self.run_one() is not None:
            done += 1
        return done

    def status(self, job_id: str) -> dict | None:
        row = self.db.one("select * from jobs where job_id = ?", (job_id,))
        if row is None:
            return None
        return {
            "job_id": row["job_id"], "kind": row["kind"], "status": row["status"],
            "attempts": row["attempts"], "max_attempts": row["max_attempts"],
            "last_error": (row["last_error"] or "")[-500:] or None,
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "run_after": row["run_after"],
        }

    def stats(self) -> dict:
        rows = self.db.query("select status, count(*) as n from jobs group by status")
        counts = {r["status"]: r["n"] for r in rows}
        now = self.clock()
        ready = self.db.one(
            "select count(*) as n from jobs where status = ? and run_after <= ?",
            (QUEUED, now))
        return {
            "queued": counts.get(QUEUED, 0),
            "ready_now": ready["n"],
            "running": counts.get(RUNNING, 0),
            "done": counts.get(DONE, 0),
            "dead": counts.get(DEAD, 0),
            "worker_id": self.worker_id,
        }

    def dead_letters(self, limit: int = 50) -> list[dict]:
        return [
            {"job_id": r["job_id"], "kind": r["kind"],
             "payload": json.loads(r["payload"]), "attempts": r["attempts"],
             "last_error": (r["last_error"] or "")[-1000:]}
            for r in self.db.query(
                "select * from jobs where status = ? order by updated_at desc limit ?",
                (DEAD, limit))
        ]

    def requeue_dead(self, job_id: str) -> bool:
        """Put a dead-lettered job back after the underlying cause is fixed."""
        cur = self.db.execute(
            "update jobs set status = ?, attempts = 0, run_after = ?, updated_at = ? "
            "where job_id = ? and status = ?",
            (QUEUED, self.clock(), self.clock(), job_id, DEAD))
        return cur.rowcount > 0


class Worker(threading.Thread):
    """Background worker thread. Poll interval is deliberately short since the
    only cost of an empty poll here is a single indexed SELECT."""

    def __init__(self, queue: JobQueue, poll_interval: float = 0.25):
        super().__init__(daemon=True, name=f"worker-{queue.worker_id}")
        self.queue = queue
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self.processed = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.queue.run_one() is not None:
                    self.processed += 1
                    continue
            except Exception:
                # a worker loop that can die on an unexpected error is a worker
                # loop that silently stops draining the queue
                traceback.print_exc()
            self._stop.wait(self.poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.join(timeout=timeout)
