"""
Serve-time repair of void open-set candidates.

A candidate whose derived name means "no identification" (e.g. the model
answered "NOT IDENTIFIABLE") was sent to logo.dev, which matched
identifiable.ca and reported it *verified*. Those records are already
persisted in finished jobs, so correcting only the derivation path in
src/openset.py would leave every existing job still displaying a fabricated
brand. `_sanitize_void_candidates` repairs them at serve time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import _sanitize_void_candidates  # noqa: E402


def _dash(*candidates):
    return {
        "title": "T",
        "open_set": {
            "available": True,
            "resolved": len(
                [c for c in candidates if c.get("status") == "candidate_verified"]
            ),
            "candidates": list(candidates),
        },
    }


def _void_candidate():
    return {
        "candidate_name": "NOT IDENTIFIABLE",
        "status": "candidate_verified",
        "frame_index": 268,
        "crop_url": "/api/crop/abc",
        "logo_dev_validation": {
            "brand": "identifiable",
            "domain": "identifiable.ca",
            "status": "verified",
        },
    }


def _real_candidate():
    return {
        "candidate_name": "TSMC (TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY LIMITED)",
        "status": "candidate_verified",
        "frame_index": 123,
        "logo_dev_validation": {
            "brand": "Taiwan Semiconductor Manufacturing Company",
            "domain": "tsmc.com",
            "status": "verified",
        },
    }


def test_void_candidate_is_downgraded_and_stops_claiming_verified():
    out = _sanitize_void_candidates(_dash(_void_candidate()))
    cand = out["open_set"]["candidates"][0]
    assert cand["status"] == "candidate_void"
    assert cand["logo_dev_validation"] is None
    # the crop and frame reference survive so the operator can still judge it
    assert cand["frame_index"] == 268
    assert cand["crop_url"] == "/api/crop/abc"
    assert out["open_set"]["resolved"] == 0


def test_resolved_count_excludes_repaired_candidates():
    out = _sanitize_void_candidates(_dash(_real_candidate(), _void_candidate()))
    statuses = [c["status"] for c in out["open_set"]["candidates"]]
    assert statuses == ["candidate_verified", "candidate_void"]
    assert out["open_set"]["resolved"] == 1


def test_real_candidates_are_untouched():
    original = _dash(_real_candidate())
    out = _sanitize_void_candidates(original)
    assert out["open_set"]["candidates"] == original["open_set"]["candidates"]


def test_sanitizer_does_not_mutate_the_stored_record():
    original = _dash(_void_candidate())
    _sanitize_void_candidates(original)
    stored = original["open_set"]["candidates"][0]
    assert stored["status"] == "candidate_verified"
    assert stored["logo_dev_validation"]["status"] == "verified"


def test_missing_or_empty_open_set_is_passed_through():
    assert _sanitize_void_candidates({"title": "T"}) == {"title": "T"}
    empty = _dash()
    assert _sanitize_void_candidates(empty) is empty


def test_pipeline_route_repairs_a_stored_void_candidate():
    """The repair is only worth anything if the served route applies it.

    The five tests above exercise the helper directly, so they all stay green if
    someone drops the call from the endpoint and the fabricated brand is served
    again. This pins the wiring.
    """
    import server

    server.JOBS.clear()
    server.JOBS["VOID1"] = {
        "job_id": "VOID1",
        "status": "done",
        "dashboard": _dash(_void_candidate()),
    }
    try:
        client = server.app.test_client()
        resp = client.get("/api/pipeline/VOID1")
        assert resp.status_code == 200
        served = resp.get_json()["open_set"]
        assert served["candidates"][0]["status"] == "candidate_void"
        assert served["candidates"][0]["logo_dev_validation"] is None
        assert served["resolved"] == 0
        # the stored record is still untouched, so the next read repairs again
        stored = server.JOBS["VOID1"]["dashboard"]["open_set"]
        assert stored["candidates"][0]["status"] == "candidate_verified"
    finally:
        server.JOBS.clear()
