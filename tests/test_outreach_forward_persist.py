"""outreach_forward must persist the mutated `forwarded` state to the store,
so a forwarded outreach survives a server restart (the process-memory-only bug)."""

import os
import tempfile
from unittest import mock

import server
from src.store import JobStore


def _fresh_store():
    tmpdir = tempfile.mkdtemp()
    return JobStore(os.path.join(tmpdir, "archive.db"))


def _fake_job():
    return {
        "job_id": "JOB1",
        "status": "done",
        "created_at": "2026-01-01T00:00:00Z",
        "dashboard": {"creator": "CHANNEL", "confidence": 0.9},
        "result": {},
    }


def test_outreach_forward_persists_forwarded_state():
    original_enabled = server.OUTREACH_ENABLED
    store = _fresh_store()
    job = _fake_job()
    server.OUTREACH_ENABLED = True
    server.JOBS["JOB1"] = job
    try:
        client = server.app.test_client()
        with mock.patch.object(server, "JOB_STORE", store):
            resp = client.post("/api/outreach/forward", json={
                "job_id": "JOB1", "brand": "NIKE",
            })
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "forwarded"

        loaded = store.load("JOB1")
        assert loaded is not None
        assert loaded["status"] == "done"
        assert loaded["forwarded"] == job["forwarded"]
        assert loaded["forwarded"][0]["brand"] == "NIKE"
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop("JOB1", None)
        store.close()
