"""SQLite storage with versioned migrations.

SQLite rather than Postgres on purpose: this needs to run from a clone with no
docker-compose, and everything below (WAL, busy timeout, transactions, upserts)
is the same shape you'd write against Postgres. Swapping the driver is a day's
work; picking the wrong concurrency model is not.

Migrations are numbered and applied in order inside a transaction, with the
applied set recorded. "Just run the CREATE TABLEs at startup" works right up
until a column needs to change on a database with data in it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        create table api_keys (
            key_id      text primary key,
            key_hash    text not null,
            name        text not null,
            scopes      text not null default 'read',
            created_at  real not null,
            revoked_at  real,
            monthly_cent_budget integer not null default 500
        );

        create table repositories (
            repo_id     text primary key,
            owner_key   text not null,
            name        text not null,
            source      text not null,
            status      text not null default 'pending',
            created_at  real not null,
            indexed_at  real,
            doc_count   integer not null default 0,
            error       text
        );

        create table documents (
            doc_id      text primary key,
            repo_id     text not null references repositories(repo_id),
            path        text not null,
            content     text not null,
            start_line  integer not null default 1,
            end_line    integer not null default 1,
            token_count integer not null default 0
        );
        create index documents_repo on documents(repo_id);

        create table jobs (
            job_id        text primary key,
            kind          text not null,
            payload       text not null,
            status        text not null default 'queued',
            attempts      integer not null default 0,
            max_attempts  integer not null default 3,
            run_after     real not null,
            locked_until  real,
            worker        text,
            created_at    real not null,
            updated_at    real not null,
            last_error    text,
            idempotency_key text unique
        );
        create index jobs_ready on jobs(status, run_after);

        create table usage (
            id          integer primary key autoincrement,
            key_id      text not null,
            request_id  text not null,
            route       text not null,
            prompt_tokens     integer not null default 0,
            completion_tokens integer not null default 0,
            embed_tokens      integer not null default 0,
            cost_micros integer not null default 0,
            cached      integer not null default 0,
            created_at  real not null
        );
        create index usage_key_time on usage(key_id, created_at);

        create table answer_cache (
            cache_key   text primary key,
            repo_id     text not null,
            question    text not null,
            answer      text not null,
            citations   text not null,
            cost_micros integer not null default 0,
            created_at  real not null,
            hits        integer not null default 0
        );
        create index answer_cache_repo on answer_cache(repo_id);
    """),
]


class Database:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._local = threading.local()
        # an in-memory database is per connection, so a threaded server would
        # see a different empty db on every thread. one shared connection then.
        self._shared = sqlite3.connect(path, check_same_thread=False) if path == ":memory:" else None
        if self._shared:
            self._configure(self._shared)
        self.migrate()

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=WAL")
        conn.execute("pragma synchronous=NORMAL")
        conn.execute("pragma foreign_keys=ON")
        # without this, a second writer fails instantly with "database is
        # locked" instead of waiting its turn
        conn.execute("pragma busy_timeout=5000")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            self._configure(conn)
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self):
        """A transaction that rolls back on any exception.

        Half-applied writes are the thing that turns a bug into a data problem.
        """
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def migrate(self) -> int:
        conn = self.conn
        conn.execute("create table if not exists schema_version "
                     "(version integer primary key, applied_at real not null)")
        conn.commit()
        applied = {r[0] for r in conn.execute("select version from schema_version")}
        run = 0
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            with self.tx() as c:
                c.executescript(sql)
                c.execute("insert into schema_version (version, applied_at) values (?, ?)",
                          (version, time.time()))
            run += 1
        return run

    @property
    def version(self) -> int:
        row = self.conn.execute("select max(version) from schema_version").fetchone()
        return row[0] or 0

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple = ()):
        with self.tx() as c:
            return c.execute(sql, params)

    def close(self) -> None:
        if self._shared:
            self._shared.close()
