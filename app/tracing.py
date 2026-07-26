"""Structured logging and request tracing.

Two rules, both learned the hard way by anyone who has debugged a production
incident from logs:

  every line is JSON with a request id
      "Error processing request" appearing 400 times in a text log tells you
      nothing. The same line with a request id lets you pull every event for
      the one request a user is complaining about.

  secrets never reach the log
      API keys arrive in headers and end up in logged context by accident. The
      redaction pass runs on every field on the way out, so it does not depend
      on the caller remembering.

Spans are a flat list with parent ids rather than a tree, which is enough to
answer "where did the time go" without pulling in an OpenTelemetry dependency.
The exporter is deliberately a seam: `Tracer.emit` is the one place to swap in
a real backend.
"""

from __future__ import annotations

import contextvars
import json
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")

# key-shaped things and anything named like a credential
SECRET_PATTERNS = [
    re.compile(r"\bcqa_[a-f0-9]{16}_[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.I),
]
SECRET_FIELDS = {"authorization", "api_key", "apikey", "key", "token",
                 "secret", "password", "x-api-key"}


def redact(value):
    if isinstance(value, str):
        out = value
        for pat in SECRET_PATTERNS:
            out = pat.sub("[redacted]", out)
        return out
    if isinstance(value, dict):
        return {k: ("[redacted]" if k.lower() in SECRET_FIELDS else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    return value


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    request_id: str
    start: float
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    def as_dict(self) -> dict:
        return {
            "span": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "request_id": self.request_id,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            **redact(self.attributes),
        }


class Tracer:
    def __init__(self, service: str = "codebase-qa", stream=None, keep: int = 2000):
        self.service = service
        self.stream = stream if stream is not None else sys.stdout
        self.spans: list[Span] = []
        self.keep = keep
        self._current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "span_id", default=None)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields) -> dict:
        """The single place a log line is produced. Swap this for a real sink."""
        line = {
            "ts": round(time.time(), 3),
            "service": self.service,
            "event": event,
            "request_id": request_id_var.get(),
            **redact(fields),
        }
        print(json.dumps(line, default=str), file=self.stream, flush=False)
        return line

    @contextmanager
    def span(self, name: str, **attributes):
        span = Span(name=name, span_id=uuid.uuid4().hex[:12],
                    parent_id=self._current.get(), request_id=request_id_var.get(),
                    start=time.perf_counter(), attributes=dict(attributes))
        token = self._current.set(span.span_id)
        try:
            yield span
        except Exception as e:
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.end = time.perf_counter()
            self._current.reset(token)
            with self._lock:
                self.spans.append(span)
                if len(self.spans) > self.keep:
                    del self.spans[: len(self.spans) - self.keep]
            self.emit("span", **span.as_dict())

    def trace(self, request_id: str) -> list[dict]:
        with self._lock:
            return [s.as_dict() for s in self.spans if s.request_id == request_id]

    def timing_summary(self) -> dict:
        """Per span-name count and p50/p95, which is what you actually want
        when asked why the endpoint got slow."""
        by_name: dict[str, list[float]] = {}
        with self._lock:
            for s in self.spans:
                by_name.setdefault(s.name, []).append(s.duration_ms)
        out = {}
        for name, xs in sorted(by_name.items()):
            xs.sort()
            out[name] = {
                "count": len(xs),
                "p50_ms": round(xs[len(xs) // 2], 3),
                "p95_ms": round(xs[min(int(len(xs) * 0.95), len(xs) - 1)], 3),
                "max_ms": round(xs[-1], 3),
            }
        return out


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
