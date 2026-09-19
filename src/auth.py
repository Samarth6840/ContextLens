"""
Phase 3 — Authentication & sessions (SaaS dashboard login).

Purposely dependency-free: password hashing uses the stdlib PBKDF2-HMAC-SHA256
with a per-user random salt and configurable iterations (no bcrypt install
needed); sessions are opaque random tokens stored in-process with a TTL.

Design principles:
  * Passwords are NEVER stored in plaintext — only `salt` + `iterations` +
    the PBKDF2 digest survive.
  * The token is a high-entropy random string handed to the client; the server
    keeps only its SHA-256 hash, so a leaked database cannot be replayed against
    a session that was never issued/copied cleanly.
  * Sessions expire after a configurable lifetime and are pruned lazily on
    access, so stale tokens do not accumulate.

Feature-flagged in the server (auth_enabled); when disabled (current default)
the dashboard/API behave exactly as before for the standalone/desktop use case.
"""

import hmac
import hashlib
import secrets
import threading
import time
from typing import Dict, Optional, Tuple

_PBKDF2_ITERATIONS = 120_000
_SALT_BYTES = 16
_DIGEST_BYTES = 32


def hash_password(password: str, iterations: int = _PBKDF2_ITERATIONS) -> Dict[str, str]:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return {
        "salt": salt.hex(),
        "iterations": str(iterations),
        "hash": digest.hex(),
    }


def verify_password(password: str, record: Dict[str, str]) -> bool:
    try:
        salt = bytes.fromhex(record["salt"])
        iterations = int(record["iterations"])
        expected = bytes.fromhex(record["hash"])
    except (KeyError, ValueError):
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(digest, expected)


class SessionManager:
    """In-memory token sessions with TTL and lazy expiry pruning."""

    def __init__(self, ttl_seconds: int = 8 * 3600):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._tokens: Dict[str, Dict[str, object]] = {}

    def create(self, user: str) -> str:
        token = secrets.token_urlsafe(32)
        digest = _token_digest(token)
        with self._lock:
            self._tokens[digest] = {
                "user": user,
                "expires_at": time.time() + self.ttl_seconds,
            }
        return token

    def validate(self, token: Optional[str]) -> Optional[str]:
        """Return the username if `token` is valid & unexpired, else None."""
        if not token:
            return None
        digest = _token_digest(token)
        now = time.time()
        with self._lock:
            rec = self._tokens.get(digest)
            if rec is None:
                return None
            if rec["expires_at"] < now:
                self._tokens.pop(digest, None)
                return None
            return str(rec["user"])

    def revoke(self, token: Optional[str]) -> None:
        if not token:
            return
        digest = _token_digest(token)
        with self._lock:
            self._tokens.pop(digest, None)

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._tokens.items()
                       if v["expires_at"] < now]
            for k in expired:
                self._tokens.pop(k, None)
            return len(expired)

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._tokens)


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def make_default_credentials(password: str) -> Tuple[str, Dict[str, str]]:
    """Convenience to create a single admin user record."""
    return "admin", hash_password(password)
