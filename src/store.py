"""
Phase 3 — SQLite-backed job store (SaaS persistence).

The server previously kept all jobs in an in-memory dict, which is lost on
restart. JobStore persists each job to SQLite (stdlib `sqlite3`, no new
dependency), so completed jobs, their dashboards, personalized recommendations,
creator profiles and outreach states survive a server restart.

Persistence is selective: the raw pipeline `result` can contain large floating
point embeddings (fusion, DINOv2) that need not be serialized. `prune_for_store`
keeps the small, high-value slices needed to render the dashboard and to drive
the Phase 3 personalized outreach generator, and drops the heavy embed payloads.

Thread-safe (a single writer lock guards all statements).
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_db_path() -> str:
    base = os.environ.get("ADSCENE_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "var"
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "contextlens.db")


class JobStore:
    """Thread-safe SQLite persistence for pipeline jobs."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    def save(self, job_id: str, job: Dict[str, Any]) -> None:
        """Persist a (JSON-serializable subset of a) job keyed by job_id."""
        payload = json.dumps(job, default=str)
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (job_id, payload, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at,
                    status=excluded.status
                """,
                (job_id, payload,
                 job.get("created_at", now), now, job.get("status")),
            )
            self._conn.commit()

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (TypeError, ValueError):
            logger.warning("Corrupt stored job %s", job_id)
            return None

    def list(self) -> List[Dict[str, Any]]:
        """Return all stored jobs as dicts (with their persisted payload)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                out.append(json.loads(row["payload"]))
            except (TypeError, ValueError):
                continue
        return out

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE job_id=?", (job_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs"
            ).fetchone()
        return int(row["n"])


def prune_for_store(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a pipeline result to the small, dashboard/outreach-relevant subset.

    Drops the heavy floating-point embed payloads (fusion fused_embed, Layer 1
    per-frame embeddings) while retaining recommendations, the creator profile,
    the cross-video brand-memory summary, and evidence/confidence data.
    """
    if not isinstance(result, dict):
        return {}
    l3 = result.get("layer3") or {}
    l2d = result.get("layer2d") or {}
    l2c = result.get("layer2c") or {}
    l2b = result.get("layer2b") or {}
    return {
        "video_path": result.get("video_path"),
        "num_frames": result.get("num_frames"),
        "video_total_frames": result.get("video_total_frames"),
        "video_fps": result.get("video_fps"),
        "has_audio": result.get("has_audio"),
        "layer2b_confidence": l2b.get("confidence"),
        "layer3": {
            "recommendations": l3.get("recommendations") or [],
        },
        "layer2d": {
            "creator_profile": l2d.get("creator_profile"),
        },
        "layer2c": {
            "brand_timeline": l2c.get("brand_timeline") or {},
            "memory_size": l2c.get("memory_size"),
            "memory_brands": l2c.get("memory_brands") or [],
            "indirect_resolutions": l2c.get("indirect_resolutions") or [],
        },
    }
