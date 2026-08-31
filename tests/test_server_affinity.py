"""Tests that the server attaches a trained affinity model to the pipeline from
stored job history, and stays cold-start (no-op) when history is absent."""

import os
import tempfile
from unittest import mock

import pytest
from src.store import JobStore

try:
    from src.layer3.affinity import CreatorBrandAffinityModel
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False

import server

pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not available"
)


def _fresh_store():
    tmpdir = tempfile.mkdtemp()
    return JobStore(os.path.join(tmpdir, "archive.db"))


def _save_job(store, job_id, creator, tallies):
    store.save(job_id, {
        "job_id": job_id,
        "status": "done",
        "creator": creator,
        "created_at": "2026-01-01T00:00:00Z",
        "dashboard": {},
        "result": {"layer2d": {"creator_profile": {
            "creator_id": creator, "brand_tallies": tallies,
        }}},
    })


def test_pipeline_gets_fitted_affinity_model_after_training():
    store = _fresh_store()
    _save_job(store, "J1", "alice", {"NIKE": 2, "ADIDAS": 1})
    _save_job(store, "J2", "alice", {"NIKE": 1})
    _save_job(store, "J3", "bob", {"SONY": 3})

    # Pipeline object without the heavy Phase1Pipeline.__init__.
    pipe = mock.Mock()
    pipe.set_affinity_model = mock.Mock()
    try:
        with mock.patch.object(server, "JOB_STORE", store):
            server._attach_affinity_model(pipe)
        assert pipe.set_affinity_model.call_count == 1
        model = pipe.set_affinity_model.call_args[0][0]
        assert model.is_fitted is True
        scores = model.predict_affinity("alice", ["NIKE", "SONY"])
        assert scores is not None and scores[0] > scores[1]
    finally:
        store.close()


def test_pipeline_stays_cold_start_without_history():
    store = _fresh_store()
    pipe = mock.Mock()
    pipe.set_affinity_model = mock.Mock()
    try:
        with mock.patch.object(server, "JOB_STORE", store):
            server._attach_affinity_model(pipe)
        assert pipe.set_affinity_model.call_count == 0
    finally:
        store.close()
