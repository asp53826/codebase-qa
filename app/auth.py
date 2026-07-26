"""API key authentication.

Three things that are easy to get wrong and expensive to get wrong:

  keys are stored hashed
      a database dump must not be a list of working credentials. Only the hash
      is stored, so the plaintext is shown exactly once at creation and is
      unrecoverable afterwards.

  lookup is by a public prefix, comparison is constant time
      hashing every key in the table on every request doesn't scale, so the key
      carries a public id to look up by. The secret half is then compared with
      compare_digest, because `==` on a hash short-circuits on the first
      differing byte and leaks a timing signal.

  revocation is a timestamp, not a delete
      "when did this key stop working" is a question you get asked during an
      incident, and a deleted row can't answer it.

The hash is scrypt: a bare sha256 of a high-entropy random key is defensible,
but keys get pasted into config files and reused, and a slow KDF costs
microseconds here and years to an attacker with the table.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

KEY_PREFIX = "cqa"
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1

VALID_SCOPES = {"read", "write", "admin"}


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class ApiKey:
    key_id: str
    name: str
    scopes: set[str]
    created_at: float
    monthly_cent_budget: int

    def can(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes


def _hash(secret: str, salt: bytes) -> str:
    digest = hashlib.scrypt(secret.encode(), salt=salt, n=SCRYPT_N,
                            r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"{salt.hex()}${digest.hex()}"


def _verify(secret: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(_hash(secret, bytes.fromhex(salt_hex)), stored)


def generate_key() -> tuple[str, str, str]:
    """Returns (full_key, key_id, secret).

    The id is the public half, safe to log and to index on. The secret half is
    never stored and never logged.
    """
    key_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    return f"{KEY_PREFIX}_{key_id}_{secret}", key_id, secret


def parse_key(raw: str) -> tuple[str, str] | None:
    parts = (raw or "").strip().split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


class AuthService:
    def __init__(self, db):
        self.db = db

    def create_key(self, name: str, scopes: set[str] | None = None,
                   monthly_cent_budget: int = 500) -> tuple[str, ApiKey]:
        scopes = scopes or {"read"}
        bad = scopes - VALID_SCOPES
        if bad:
            raise ValueError(f"unknown scope(s): {', '.join(sorted(bad))}")

        full, key_id, secret = generate_key()
        record = ApiKey(key_id, name, scopes, time.time(), monthly_cent_budget)
        self.db.execute(
            "insert into api_keys (key_id, key_hash, name, scopes, created_at, "
            "monthly_cent_budget) values (?, ?, ?, ?, ?, ?)",
            (key_id, _hash(secret, secrets.token_bytes(16)), name,
             ",".join(sorted(scopes)), record.created_at, monthly_cent_budget),
        )
        # the only time the caller ever sees the plaintext
        return full, record

    def authenticate(self, raw: str | None) -> ApiKey:
        parsed = parse_key(raw or "")
        if not parsed:
            raise AuthError("missing or malformed API key")
        key_id, secret = parsed

        row = self.db.one("select * from api_keys where key_id = ?", (key_id,))
        if row is None:
            # same cost and same message as a wrong secret, so an unknown id
            # can't be distinguished from a bad one
            _verify(secret, _hash("decoy", b"\x00" * 16))
            raise AuthError("invalid API key")
        if not _verify(secret, row["key_hash"]):
            raise AuthError("invalid API key")
        if row["revoked_at"] is not None:
            raise AuthError("API key has been revoked")

        return ApiKey(row["key_id"], row["name"], set(row["scopes"].split(",")),
                      row["created_at"], row["monthly_cent_budget"])

    def require(self, raw: str | None, scope: str) -> ApiKey:
        key = self.authenticate(raw)
        if not key.can(scope):
            raise AuthError(
                f"this key has scope(s) {','.join(sorted(key.scopes))} and the "
                f"route requires {scope}", status=403)
        return key

    def revoke(self, key_id: str) -> bool:
        cur = self.db.execute(
            "update api_keys set revoked_at = ? where key_id = ? and revoked_at is null",
            (time.time(), key_id))
        return cur.rowcount > 0

    def rotate(self, key_id: str) -> tuple[str, ApiKey]:
        """Issue a replacement and revoke the old one, keeping name and budget."""
        row = self.db.one("select * from api_keys where key_id = ?", (key_id,))
        if row is None:
            raise AuthError("no such key", status=404)
        full, record = self.create_key(row["name"], set(row["scopes"].split(",")),
                                       row["monthly_cent_budget"])
        self.revoke(key_id)
        return full, record

    def list_keys(self) -> list[dict]:
        return [
            {"key_id": r["key_id"], "name": r["name"], "scopes": r["scopes"],
             "created_at": r["created_at"], "revoked": r["revoked_at"] is not None,
             "monthly_cent_budget": r["monthly_cent_budget"]}
            for r in self.db.query("select * from api_keys order by created_at")
        ]
