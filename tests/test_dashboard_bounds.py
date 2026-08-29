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

from server import _validate_dashboard_bounds  # noqa: E402


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
