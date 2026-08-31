"""Tests for /api/insights/<id> falling back to the SQLite store when a job is
not in live memory (restarted / seeded data)."""

import os
import tempfile
from unittest import mock

import server
from src.store import JobStore


def _fresh_store():
    tmpdir = tempfile.mkdtemp()
    return JobStore(os.path.join(tmpdir, "archive.db"))


def test_insights_falls_back_to_archive_job():
    store = _fresh_store()
    store.save("ARCHINS1", {
        "job_id": "ARCHINS1",
        "status": "done",
        "creator": "seed_demo",
        "created_at": "2026-01-01T00:00:00Z",
        "dashboard": {},
        "result": {
            "layer3": {"recommendations": [{"brand": "NIKE", "type": "DIRECT",
                                            "score": 0.9, "reasons": ["X"]}]},
            "layer2d": {"creator_profile": {
                "creator_id": "seed_demo", "handle": "demo",
                "brand_tallies": {"NIKE": 2}, "categories": {"APPAREL": 1},
            }},
            "layer2c": {
                "memory_size": 1, "memory_brands": ["NIKE"],
                "indirect_resolutions": [{"brand": "NIKE", "reference": "this shoe"}],
            },
        },
    })
    client = server.app.test_client()
    try:
        with mock.patch.object(server, "JOB_STORE", store), \
             mock.patch.object(server, "JOBS", {}):
            resp = client.get("/api/insights/ARCHINS1")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["recommendations"][0]["brand"] == "NIKE"
        assert body["creator_profile"]["handle"] == "demo"
        assert body["brand_memory"]["size"] == 1
        assert body["brand_memory"]["indirect_resolutions"][0]["brand"] == "NIKE"
    finally:
        store.close()


def test_insights_missing_job_returns_409():
    store = _fresh_store()
    client = server.app.test_client()
    try:
        with mock.patch.object(server, "JOB_STORE", store), \
             mock.patch.object(server, "JOBS", {}):
            resp = client.get("/api/insights/NOPE")
        assert resp.status_code == 409
        assert "NOT AVAILABLE" in resp.get_json()["error"]
    finally:
        store.close()
