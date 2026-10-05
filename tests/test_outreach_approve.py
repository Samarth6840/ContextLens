"""The outreach approval endpoint must persist AND must not claim to send.

It used to be `/api/outreach/forward`, returning {"status": "forwarded"} while
only appending a timestamp to the job record. The UI rendered "Forwarded to
<creator>". Nothing was ever sent, so the tool lied about its own side effect
on the same page it is meant to keep honest.
"""

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


def _post(client, **body):
    return client.post("/api/outreach/approve", json=body)


def test_approval_persists_across_a_server_restart():
    original_enabled = server.OUTREACH_ENABLED
    store = _fresh_store()
    job = _fake_job()
    server.OUTREACH_ENABLED = True
    server.JOBS["JOB1"] = job
    try:
        client = server.app.test_client()
        with mock.patch.object(server, "JOB_STORE", store):
            resp = _post(client, job_id="JOB1", brand="NIKE")
        assert resp.status_code == 200

        loaded = store.load("JOB1")
        assert loaded is not None
        assert loaded["status"] == "done"
        assert loaded["approved"] == job["approved"]
        assert loaded["approved"][0]["brand"] == "NIKE"
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop("JOB1", None)
        store.close()


def test_response_never_claims_something_was_sent():
    """The regression that mattered: "forwarded" implies a send that never
    happened. `sent: False` and a non-"forwarded" status are both required —
    a future client reading either one must not display a delivery claim."""
    original_enabled = server.OUTREACH_ENABLED
    store = _fresh_store()
    job = _fake_job()
    server.OUTREACH_ENABLED = True
    server.JOBS["JOB1"] = job
    try:
        client = server.app.test_client()
        with mock.patch.object(server, "JOB_STORE", store):
            resp = _post(client, job_id="JOB1", brand="NIKE")
        body = resp.get_json()
        assert body["status"] != "forwarded"
        assert body["status"] == "marked_approved"
        assert body["sent"] is False
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop("JOB1", None)
        store.close()


def test_approval_requires_a_brand():
    original_enabled = server.OUTREACH_ENABLED
    store = _fresh_store()
    job = _fake_job()
    server.OUTREACH_ENABLED = True
    server.JOBS["JOB1"] = job
    try:
        client = server.app.test_client()
        with mock.patch.object(server, "JOB_STORE", store):
            resp = _post(client, job_id="JOB1", brand=None)
        assert resp.status_code == 400
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop("JOB1", None)
        store.close()


def test_the_faking_endpoint_is_gone():
    """No route may still accept /api/outreach/forward."""
    routes = {r.rule for r in server.app.url_map.iter_rules()}
    assert "/api/outreach/forward" not in routes
