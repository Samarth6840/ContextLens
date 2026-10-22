"""Active-learning frame selection: label only what the system gets wrong."""

import importlib.util as ilu
import json
import sys
from pathlib import Path

_spec = ilu.spec_from_file_location(
    "collect", Path(__file__).resolve().parent.parent / "scripts"
    / "collect_active_learning_frames.py"
)
collect = ilu.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def _result():
    return {
        "video_fps": 1.0,
        "video_stride": 1,
        "num_frames": 10,
        "layer1": {
            # frame 0: box but nothing resolved -> hard negative
            # frame 1: resolved -> skip
            # frame 2: no boxes
            "logo_detections": [
                [{"brand": None, "confidence": 0.31}],
                [{"brand": "NIKE", "confidence": 0.9}],
                [],
            ],
            # NIKE spoken at t=3s -> frame 3 has no box -> missed detection
            "brand_mentions": [{"brand": "NIKE", "start_time": 3.0}],
        },
    }


def test_selects_unresolved_and_missed_frames():
    cands = collect.frame_candidates(_result())
    assert {(c["frame_index"], c["reason"]) for c in cands} == {
        (0, "unresolved_logo"),
        (3, "missed_detection"),
    }


def test_resolved_frames_are_skipped():
    cands = collect.frame_candidates(_result())
    assert all(c["frame_index"] != 1 for c in cands)


def test_mention_on_a_frame_with_boxes_is_not_a_miss():
    r = _result()
    # Move the mention to t=1s (frame 1 has a box) -> no missed_detection.
    r["layer1"]["brand_mentions"] = [{"brand": "NIKE", "start_time": 1.0}]
    cands = collect.frame_candidates(r)
    assert all(c["reason"] != "missed_detection" for c in cands)


def test_dedupes_repeat_mentions_to_one_row():
    r = _result()
    r["layer1"]["brand_mentions"] = [
        {"brand": "NIKE", "start_time": 3.0},
        {"brand": "NIKE", "start_time": 3.0},
    ]
    cands = collect.frame_candidates(r)
    missed = [c for c in cands if c["reason"] == "missed_detection"]
    assert len(missed) == 1


def test_empty_result_is_empty():
    assert collect.frame_candidates({}) == []
    assert collect.frame_candidates({"layer1": {}}) == []


def _dashboard():
    return {
        "scenes": [
            {"frame_index": 0, "logos": [
                {"class_name": "UNKNOWN BRAND", "confidence": 0.3}]},
            {"frame_index": 1, "logos": [
                {"class_name": "NIKE", "confidence": 0.9}]},
            {"frame_index": 2, "logos": []},
        ],
        "ads": [
            {"brand": "ADIDAS", "frame_index": 5},
            {"brand": "NIKE", "frame_index": 1},  # resolved frame -> not a miss
        ],
    }


def test_dashboard_candidates_from_persisted_snapshot():
    c = collect.dashboard_candidates(_dashboard())
    keys = {(x["frame_index"], x["reason"]) for x in c}
    assert keys == {(0, "unresolved_logo"), (5, "missed_detection")}


def _tiny_video(path: Path, n_frames: int = 8, size=(32, 32)) -> Path:
    """A real mp4 whose frame i has gray level i*30, so we can check the seek."""
    import cv2
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    assert w.isOpened()
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), i * 30, np.uint8))
    w.release()
    return path


def test_extract_frames_writes_named_jpegs_at_source_indices(tmp_path):
    vid = _tiny_video(tmp_path / "src.mp4")
    out = tmp_path / "frames"
    rows = [
        {"frame_index": 2, "reason": "unresolved_logo"},
        {"frame_index": 5, "reason": "missed_detection"},
    ]
    counts = collect.extract_frames(str(vid), 1, rows, out, "job1")
    assert counts == {"written": 2, "missing_video": 0, "read_failed": 0}
    assert (out / "job1_f000002_unresolved_logo.jpg").is_file()
    assert (out / "job1_f000005_missed_detection.jpg").is_file()


def test_extract_frames_maps_sampled_index_through_stride(tmp_path):
    """Sampled index 2 with stride 3 must be source frame 6 (gray 180)."""
    import cv2

    vid = _tiny_video(tmp_path / "src.mp4", n_frames=9)
    out = tmp_path / "frames"
    counts = collect.extract_frames(
        str(vid), 3, [{"frame_index": 2, "reason": "unresolved_logo"}],
        out, "job1")
    assert counts["written"] == 1
    got = cv2.imread(str(out / "job1_f000002_unresolved_logo.jpg"),
                     cv2.IMREAD_GRAYSCALE)
    assert abs(int(got.mean()) - 180) < 25  # lossy codec, tight enough to prove seek


def test_extract_frames_reports_missing_video_instead_of_crashing(tmp_path):
    rows = [{"frame_index": 0, "reason": "unresolved_logo"},
            {"frame_index": 1, "reason": "missed_detection"}]
    counts = collect.extract_frames(str(tmp_path / "gone.mp4"), 1, rows,
                                    tmp_path / "frames", "job1")
    assert counts == {"written": 0, "missing_video": 2, "read_failed": 0}
    assert not (tmp_path / "frames").exists()


def test_extract_frames_no_rows_is_a_noop(tmp_path):
    counts = collect.extract_frames("/nonexistent.mp4", 1, [],
                                    tmp_path / "frames", "job1")
    assert counts == {"written": 0, "missing_video": 0, "read_failed": 0}
    assert not (tmp_path / "frames").exists()


def test_safe_prefix_has_no_underscore_for_only_video(tmp_path):
    # label_frames.py pins queues on name.split("_")[0].
    assert "_" not in collect._safe_prefix("job_1/odd id")
    assert collect._safe_prefix("job_1/odd id")


def test_main_extracts_frames_end_to_end(tmp_path, monkeypatch, capsys):
    vid = _tiny_video(tmp_path / "src.mp4")
    results = tmp_path / "results"
    results.mkdir()
    (results / "abc.json").write_text(json.dumps({
        "video_path": str(vid),
        "video_stride": 1,
        "layer1": {
            "logo_detections": [[{"brand": None, "confidence": 0.4}], []],
            "brand_mentions": [],
        },
    }))
    manifest = tmp_path / "al.jsonl"
    frames = tmp_path / "frames"
    monkeypatch.setattr(sys, "argv", [
        "collect_active_learning_frames.py",
        "--results", str(results), "--out", str(manifest),
        "--extract-frames", str(frames),
    ])
    assert collect.main() == 0
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x]
    assert [(r["job_id"], r["frame_index"], r["reason"]) for r in rows] == [
        ("abc", 0, "unresolved_logo")]
    assert (frames / "abc_f000000_unresolved_logo.jpg").is_file()
    assert "extracted 1 frame(s)" in capsys.readouterr().out


def test_main_extract_fails_loudly_when_no_video_exists(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "abc.json").write_text(json.dumps({
        "video_path": str(tmp_path / "gone.mp4"),
        "layer1": {"logo_detections": [[{"brand": None, "confidence": 0.4}]],
                   "brand_mentions": []},
    }))
    monkeypatch.setattr(sys, "argv", [
        "collect_active_learning_frames.py",
        "--results", str(results), "--out", str(tmp_path / "al.jsonl"),
        "--extract-frames", str(tmp_path / "frames"),
    ])
    assert collect.main() == 1


def test_snapshot_prefers_layer1_then_dashboard():
    # A pruned persisted snapshot (no layer1) falls back to the dashboard.
    snap = {"result": {"video_path": "v.mp4"}, "dashboard": _dashboard()}
    assert collect.snapshot_candidates(snap) == collect.dashboard_candidates(
        _dashboard())
    # A raw result dict uses layer1 directly.
    raw = {"layer1": {"logo_detections": [[{"brand": None, "confidence": 0.2}]],
                      "brand_mentions": []}}
    assert [(x["frame_index"], x["reason"])
            for x in collect.snapshot_candidates(raw)] == [(0, "unresolved_logo")]
