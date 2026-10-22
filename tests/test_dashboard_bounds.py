"""
Tests for the dashboard bounds assertion (escalation Task 2.2).

This is the recurrence-proof hard stop for the escalated failure mode: a video
sampled to `num_frames` frames can never again render "SCENE n" for n >
num_frames (e.g. "SCENE 059" on a video that had 3 sampled frames). The
validator must raise loudly instead of silently trimming.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (  # noqa: E402
    _enrich_recommendations,
    _scene_logo_overlays,
    _validate_dashboard_bounds,
)


def test_saturated_scene_preview_hides_all_untrusted_logo_boxes():
    logos = [{"bbox": [i, i, i + 20, i + 20], "brand": None, "confidence": 0.9}
             for i in range(100)]
    logos[0].update(brand="OPPO", resolution_quality=0.7)
    logos[1].update(brand="SAMSUNG", resolution_quality=0.9)

    visible, hidden = _scene_logo_overlays(logos)

    assert visible == []
    assert hidden == 100


def test_normal_scene_preview_keeps_all_logo_detections():
    logos = [{"bbox": [i, i, i + 20, i + 20], "brand": None} for i in range(20)]
    visible, hidden = _scene_logo_overlays(logos)
    assert visible == logos
    assert hidden == 0


def test_scene_endpoint_shows_clean_preview_and_serves_original_frame(monkeypatch):
    import numpy as np
    import server

    job_id = "OVERLAY-SATURATION-TEST"
    detections = [
        {"bbox": [i, i, i + 30, i + 30], "confidence": 0.5, "brand": None}
        for i in range(25)
    ]
    job = {
        "status": "done",
        "video_path": "unused-in-test.mp4",
        "result": {
            "num_frames": 1,
            "video_stride": 1,
            "layer1": {
                "scene_object_detections": [[]],
                "logo_detections": [detections],
            },
        },
    }
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    monkeypatch.setattr(server, "AUTH_ENABLED", False)
    monkeypatch.setattr(server, "_job", lambda _job_id: job)
    monkeypatch.setattr(server, "_read_video_frame", lambda *_args: frame.copy())
    client = server.app.test_client()

    preview = client.get(f"/api/scene/{job_id}/0")
    original = client.get(f"/api/scene/{job_id}/0?raw=1")

    assert preview.status_code == original.status_code == 200
    assert preview.data != original.data


def test_archived_pipeline_and_scene_preview_survive_server_restart(monkeypatch):
    import numpy as np
    import server
    from types import SimpleNamespace

    job = {
        "status": "done",
        "dashboard": {
            "job_id": "ARCHIVED-OVERLAY-TEST",
            "num_frames": 3,
            "video_fps": 25,
            "scenes": [{"frame_index": 1, "timestamp": 2.0}],
        },
        "result": {
            "num_frames": 3,
            "video_total_frames": 75,
            "video_fps": 25,
            "video_path": "archived-video.mp4",
        },
    }
    seeks = []
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    monkeypatch.setattr(server, "AUTH_ENABLED", False)
    monkeypatch.setattr(server, "_job", lambda _job_id: None)
    monkeypatch.setattr(server, "JOB_STORE", SimpleNamespace(load=lambda _job_id: job))
    monkeypatch.setattr(
        server, "_read_video_frame",
        lambda path, index, stride: (seeks.append((path, index, stride)) or frame.copy()),
    )
    client = server.app.test_client()

    dashboard = client.get("/api/pipeline/ARCHIVED-OVERLAY-TEST")
    image = client.get("/api/scene/ARCHIVED-OVERLAY-TEST/1")

    assert dashboard.status_code == 200
    assert image.status_code == 200
    assert seeks == [("archived-video.mp4", 50, 1)]


def _dashboard(num_frames, scenes, products=None, ads=None, open_set=None):
    return {
        "num_frames": num_frames,
        "scenes": scenes,
        "products": products or [],
        "ads": ads or [],
        "open_set": open_set or {"candidates": []},
    }


def test_scene_above_num_frames_raises():
    """SCENE 059 on a 3-frame video must raise DASHBOARD BOUND VIOLATION."""
    dash = _dashboard(
        num_frames=3,
        scenes=[{"n": 59, "frame_index": 2}],
    )
    with pytest.raises(RuntimeError, match="DASHBOARD BOUND VIOLATION"):
        _validate_dashboard_bounds(dash, 3)


def test_scene_frame_index_out_of_range_raises():
    dash = _dashboard(
        num_frames=3,
        scenes=[{"n": 1, "frame_index": 5}],
    )
    with pytest.raises(RuntimeError, match="DASHBOARD BOUND VIOLATION"):
        _validate_dashboard_bounds(dash, 3)


def test_too_many_scenes_raises():
    dash = _dashboard(num_frames=2, scenes=[{"n": 1}, {"n": 2}, {"n": 3}])
    with pytest.raises(RuntimeError, match="DASHBOARD BOUND VIOLATION"):
        _validate_dashboard_bounds(dash, 2)


def test_open_set_candidate_out_of_range_raises():
    dash = _dashboard(
        num_frames=3,
        scenes=[],
        open_set={"candidates": [{"frame_index": 60, "candidate_name": "X"}]},
    )
    with pytest.raises(RuntimeError, match="DASHBOARD BOUND VIOLATION"):
        _validate_dashboard_bounds(dash, 3)


def test_valid_dashboard_passes_unchanged():
    dash = _dashboard(
        num_frames=60,
        scenes=[{"n": 23, "frame_index": 54}],
        products=[{"brand": "B", "appearances": ["SCENE 023"]}],
        ads=[{"brand": "A", "scenes": ["FRAME 30"]}],
        open_set={"candidates": [{"frame_index": 54}]},
    )
    out = _validate_dashboard_bounds(dash, 60)
    assert out is dash


def test_enrich_recommendations_never_invents_a_contact_email():
    """The catalog used to carry guessed addresses (partnerships@nike.com,
    pr@reebok.com). Those were never sourced, so a recommendation now carries
    no email unless the Gemini grounding lookup actually found one."""
    recs = [{"brand": "NIKE", "type": "DIRECT", "score": 0.9}]
    out = _enrich_recommendations(recs)
    assert out[0]["contact_email"] is None
    assert out[0]["contact_verified"] is False
    # The curated website IS safe to show — keep it.
    assert out[0]["contact_website"] == "https://www.nike.com"
    assert out[0]["appearances"] == 0
    # Original fields preserved, not mutated.
    assert out[0]["type"] == "DIRECT" and out[0]["score"] == 0.9


def test_enrich_recommendations_preserves_existing_appearances():
    recs = [{"brand": "NIKE", "appearances": 4}]
    out = _enrich_recommendations(recs)
    assert out[0]["appearances"] == 4  # not overwritten to 0


def test_enrich_recommendations_handles_unknown_and_malformed():
    out = _enrich_recommendations([
        {"brand": "NOT_A_CATALOG_BRAND"},
        "junk",
        None,
    ])
    assert out[0]["contact_email"] is None
    assert out[0]["appearances"] == 0
    assert "junk" in out and None in out
