"""
Tests for the draft-email data path fixes:

  * products now carry `appearance_count` (the draft gateway read a field that
    was never set, so every real brand was refused "NO REAL ON-SCREEN
    APPEARANCES");
  * `_enrich_recommendations` carries real appearance counts + optional
    gemini-grounded emails into the outreach editor.
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import server  # noqa: E402
from server import _build_dashboard, _enrich_recommendations  # noqa: E402


def _make_result(logos):
    framed = [list(logos)] + [[] for _ in range(3)]
    return {
        "num_frames": 4,
        "video_fps": 25.0,
        "video_total_frames": 100,
        "video_stride": 1,
        "has_audio": True,
        "layer1": {
            "logo_detections": framed,
            "ocr_results": [[{"text": "SAMSUNG", "confidence": 0.9}]] + [[] for _ in range(3)],
            "scene_object_detections": [[] for _ in range(4)],
            "transcript": "",
            "audio_events": [],
            "brand_mentions": [{"brand": "SAMSUNG", "start_time": 0.0, "end_time": 1.0}],
        },
        "layer2b": {
            "confidence": 0.9,
            "is_confident": True,
            "status": "confident",
            "evidence_breakdown": {},
        },
        "layer2c": {"unknown_brand_regions": []},
        "layer3": {"recommendations": [{"brand": "SAMSUNG", "score": 0.9, "reasons": []}]},
    }


def _logo(brand):
    return {
        "class_name": brand,
        "brand": brand,
        "confidence": 0.9,
        "resolution_quality": 0.9,
        "ocr_text": brand,
    }


def test_products_carry_appearance_count():
    result = _make_result([_logo("SAMSUNG")])
    result["layer1"]["logo_detections"] = [
        [_logo("SAMSUNG")], [_logo("SAMSUNG")], [], [],
    ]
    dash = _build_dashboard(result, {"job_id": "J1", "title": "t", "creator": "c"})
    samsung = next(p for p in dash["products"] if p["brand"] == "SAMSUNG")
    assert samsung["appearance_count"] == 2
    assert samsung["appearances"] == [0, 1]


def test_draft_email_uses_appearance_count():
    original_enabled = server.OUTREACH_ENABLED
    server.OUTREACH_ENABLED = True
    job_id = "TJ-DRAFT-1"
    server.JOBS[job_id] = {
        "job_id": job_id,
        "result": None,  # force the legacy fallback path
        "dashboard": {
            "creator": "CHANNEL",
            "title": "T",
            "products": [{
                "brand": "SAMSUNG",
                "product": "Samsung Galaxy",
                "category": "ELECTRONICS",
                "appearances": [2],
                "appearance_count": 1,
                "contact_email": "pr@samsung.com",
            }],
            "recommendations": [],
        },
    }
    try:
        client = server.app.test_client()
        with mock.patch.object(
            server, "_validate_brand", return_value={"status": "verified"}
        ):
            resp = client.post("/api/outreach/generate", json={
                "job_id": job_id, "brand": "SAMSUNG", "target": "pr@samsung.com",
            })
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert "Samsung Galaxy" in body["body"]
        assert "SAMSUNG" in body["subject"]
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop(job_id, None)


def test_draft_email_legacy_fallback_grep_behind_list_appearances():
    original_enabled = server.OUTREACH_ENABLED
    server.OUTREACH_ENABLED = True
    job_id = "TJ-DRAFT-2"
    server.JOBS[job_id] = {
        "job_id": job_id,
        "result": None,
        "dashboard": {
            "creator": "CHANNEL",
            "title": "T",
            "products": [{
                "brand": "GOOGLE",
                "product": "Pixel",
                "category": "ELECTRONICS",
                # A legacy payload with only the list — no appearance_count.
                "appearances": [1, 4, 7],
            }],
            "recommendations": [],
        },
    }
    try:
        client = server.app.test_client()
        with mock.patch.object(
            server, "_validate_brand", return_value={"status": "verified"}
        ):
            resp = client.post("/api/outreach/generate", json={
                "job_id": job_id, "brand": "GOOGLE", "target": "press@google.com",
            })
        assert resp.status_code == 200, resp.get_json()
    finally:
        server.OUTREACH_ENABLED = original_enabled
        server.JOBS.pop(job_id, None)


def test_enrichment_carries_appearance_counts_and_emails():
    recs = [
        {"brand": "SAMSUNG", "score": 0.9, "reasons": []},
        {"brand": "AMAZON", "score": 0.4, "reasons": []},
    ]
    enriched = _enrich_recommendations(
        recs,
        appearance_counts={"samsung": 3},
        emails={
            "SAMSUNG": {
                "status": "ok",
                "emails": [
                    {"email": "press@samsung.com", "type": "press", "source": "https://samsung.com/press"},
                    {"email": "hr.in@samsung.com", "type": "hr", "source": "https://samsung.com/contact"},
                ],
                "evidence": [{"url": "https://samsung.com/press"}],
            },
        },
    )
    samsung = next(r for r in enriched if r["brand"] == "SAMSUNG")
    amazon = next(r for r in enriched if r["brand"] == "AMAZON")
    assert samsung["appearances"] == 3
    assert samsung["contact_email"] == "press@samsung.com"
    assert samsung["contact_email_source"] == "gemini_grounding"
    assert samsung["contact_verified"] is False
    assert samsung["hr_emails"] == ["hr.in@samsung.com"]
    assert amazon["appearances"] == 0
    assert "contact_email_source" not in amazon