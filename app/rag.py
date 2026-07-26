"""Indexing and answering, with a cache in front.

The retrieval here is intentionally modest — BM25 over code chunks split at
function and class boundaries. The interesting work in this repo is everything
around it, and rag-eval is where the retrieval quality question is answered
properly.

The cache is the part worth reading. Two things it gets right:

  the key includes the index version
      caching on the question alone means a repo that was just re-indexed keeps
      serving answers built from the old code. The key is
      (repo, indexed_at, question), so re-indexing invalidates everything for
      that repo without a sweep.

  a cache hit costs nothing and is recorded as such
      the usage row still gets written, with cost zero and cached=1, so the hit
      rate is visible and a cached request never bills the caller.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# split camelCase and snake_case so a search for "parse" finds parseConfig
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOP = frozenset("""the a an and or if else for while return def class import from
as is in not none true false self this that with try except pass""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for word in WORD.findall(text):
        parts = CAMEL.sub(" ", word).replace("_", " ").lower().split()
        out.append(word.lower())
        if len(parts) > 1:
            out.extend(parts)
    return [t for t in out if t not in STOP and len(t) > 1]


@dataclass
class Chunk:
    path: str
    content: str
    start_line: int
    end_line: int

    @property
    def token_count(self) -> int:
        return max(1, len(self.content) // 4)


def chunk_source(path: str, source: str, max_lines: int = 60) -> list[Chunk]:
    """Split at top-level def/class boundaries, falling back to fixed windows.

    Splitting mid-function is the single most common way a code Q&A tool ends
    up unable to answer anything: the signature lands in one chunk and the body
    in another, so neither matches the question.
    """
    lines = source.splitlines()
    if not lines:
        return []

    boundaries = [i for i, ln in enumerate(lines)
                  if re.match(r"^(def |class |async def |func |function |public |private )", ln)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    chunks: list[Chunk] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        # a very long function still gets split, or one chunk swamps the index
        for s in range(start, end, max_lines):
            e = min(s + max_lines, end)
            body = "\n".join(lines[s:e]).strip()
            if body:
                chunks.append(Chunk(path, "\n".join(lines[s:e]), s + 1, e))
    return chunks


class BM25Index:
    """Rebuilt per query set rather than persisted.

    Fine at this scale and it keeps the index derivable from the documents
    table, so there is no second source of truth to fall out of sync.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, docs: list[tuple[str, str]]) -> "BM25Index":
        self.doc_ids = [d[0] for d in docs]
        self.lengths = []
        raw: dict[str, list[tuple[int, int]]] = {}
        for i, (_, text) in enumerate(docs):
            toks = tokenize(text)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                raw.setdefault(term, []).append((i, tf))
        n = len(docs)
        self.avgdl = sum(self.lengths) / n if n else 0.0
        self.postings = raw
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in raw.items()}
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, tf in posting:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RagService:
    def __init__(self, db, tracer, model: str = "local"):
        self.db = db
        self.tracer = tracer
        self.model = model

    # ---------------------------------------------------------------- index

    def index_repo(self, repo_id: str, files: dict[str, str]) -> int:
        with self.tracer.span("index_repo", repo_id=repo_id, files=len(files)):
            with self.db.tx() as c:
                c.execute("delete from documents where repo_id = ?", (repo_id,))
                total = 0
                for path, source in sorted(files.items()):
                    for chunk in chunk_source(path, source):
                        c.execute(
                            "insert into documents (doc_id, repo_id, path, content, "
                            "start_line, end_line, token_count) values (?, ?, ?, ?, ?, ?, ?)",
                            (uuid.uuid4().hex, repo_id, chunk.path, chunk.content,
                             chunk.start_line, chunk.end_line, chunk.token_count))
                        total += 1
                c.execute(
                    "update repositories set status = 'ready', indexed_at = ?, "
                    "doc_count = ?, error = null where repo_id = ?",
                    (time.time(), total, repo_id))
            # re-indexing changes indexed_at, which is in the cache key, so
            # every cached answer for this repo is invalidated implicitly
            return total

    def mark_failed(self, repo_id: str, error: str) -> None:
        self.db.execute("update repositories set status = 'failed', error = ? "
                        "where repo_id = ?", (error[:2000], repo_id))

    # ---------------------------------------------------------------- query

    def _index_version(self, repo_id: str) -> float:
        row = self.db.one("select indexed_at from repositories where repo_id = ?",
                          (repo_id,))
        return (row["indexed_at"] if row and row["indexed_at"] else 0.0)

    def cache_key(self, repo_id: str, question: str) -> str:
        blob = f"{repo_id}|{self._index_version(repo_id)}|{question.strip().lower()}"
        return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    def lookup_cache(self, repo_id: str, question: str) -> Answer | None:
        import json

        key = self.cache_key(repo_id, question)
        row = self.db.one("select * from answer_cache where cache_key = ?", (key,))
        if row is None:
            return None
        self.db.execute("update answer_cache set hits = hits + 1 where cache_key = ?",
                        (key,))
        return Answer(row["answer"], json.loads(row["citations"]), cached=True)

    def store_cache(self, repo_id: str, question: str, answer: Answer,
                    cost_micros: int) -> None:
        import json

        self.db.execute(
            "insert into answer_cache (cache_key, repo_id, question, answer, "
            "citations, cost_micros, created_at) values (?, ?, ?, ?, ?, ?, ?) "
            "on conflict(cache_key) do nothing",
            (self.cache_key(repo_id, question), repo_id, question, answer.text,
             json.dumps(answer.citations), cost_micros, time.time()))

    def retrieve(self, repo_id: str, question: str, k: int = 5) -> list[dict]:
        with self.tracer.span("retrieve", repo_id=repo_id, k=k) as span:
            rows = self.db.query(
                "select doc_id, path, content, start_line, end_line "
                "from documents where repo_id = ?", (repo_id,))
            if not rows:
                span.attributes["hits"] = 0
                return []
            by_id = {r["doc_id"]: r for r in rows}
            index = BM25Index().build([(r["doc_id"], f"{r['path']}\n{r['content']}")
                                       for r in rows])
            hits = index.search(question, k)
            span.attributes["hits"] = len(hits)
            span.attributes["corpus"] = len(rows)
            return [{
                "doc_id": d, "path": by_id[d]["path"],
                "content": by_id[d]["content"],
                "start_line": by_id[d]["start_line"],
                "end_line": by_id[d]["end_line"],
                "score": round(score, 4),
            } for d, score in hits]

    def answer(self, repo_id: str, question: str, k: int = 5) -> Answer:
        """Extractive by default: quote the retrieved code with citations.

        No LLM in the default path, so the whole service runs from a clone with
        no API key. An LLM generator drops in here without touching auth, rate
        limiting, the queue, the cache or cost tracking, which is the point of
        keeping them separate.
        """
        with self.tracer.span("answer", repo_id=repo_id):
            hits = self.retrieve(repo_id, question, k)
            if not hits:
                return Answer("I couldn't find anything relevant in this repository.",
                              [], prompt_tokens=len(question) // 4)

            citations = [{"path": h["path"], "start_line": h["start_line"],
                          "end_line": h["end_line"], "score": h["score"]}
                         for h in hits]
            parts = [f"Found {len(hits)} relevant location(s):", ""]
            for h in hits:
                parts.append(f"{h['path']}:{h['start_line']}-{h['end_line']}")
                parts.append("```")
                parts.append(h["content"][:800])
                parts.append("```")
                parts.append("")
            text = "\n".join(parts).strip()
            prompt_tokens = sum(h["content"].__len__() for h in hits) // 4
            return Answer(text, citations, prompt_tokens=prompt_tokens,
                          completion_tokens=len(text) // 4)
