import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _iou(a, b):
    from scripts.eval_logo import _iou
    return _iou(a, b)


def test_grader_hero_designation_not_largest(tmp_path):
    """Top-1 must hit the DESIGNATED hero box, never argmax-area.

    Every hero-designated logo frame carries a small hero (80,40,120,60) and a
    larger non-overlapping billboard (0,60,199,99); the frozen dets hit the
    hero box. A grader falling back to largest-by-area scores 0.0
    (iou(hero,bill)=0). One logo frame is billboard-only (no hero): it counts
    in n_logo, is reported as n_logo_no_hero, and drops out of the top-1 pool.
    """
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_grader_dryrun.py"),
         "--root", str(tmp_path), "--fps-per-free", "3"],
        capture_output=True, text=True, check=True)
    assert _iou([80, 40, 120, 60], [0, 60, 199, 99]) == 0.0
    g = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "owlv2_1p7_grade.py"),
         "--export", str(tmp_path),
         "--yolo-dets", str(tmp_path / "realvideo_v1_yolo.json"),
         "--owlv2-dets", str(tmp_path / "realvideo_v1_owlv2.json")],
        capture_output=True, text=True, check=True)
    out = json.loads(
        (Path(str(tmp_path)) / "realvideo_v1_yolo.grade.json").read_text())
    best = out["chosen_owlv2_conf"]
    assert out["pass"] is True
    assert out["fp_margin_frames"] == 12
    m = out["pooled"]["owlv2"][str(best)]
    assert m["n_logo"] == 8
    assert m["n_hero"] == 7          # one billboard-only frame has no hero
    assert m["n_logo_no_hero"] == 1
    assert m["top1_hit_rate"] == 1.0
    assert m["hits"] == 7  # hero hits, NOT the larger billboard boxes
    assert "with hero, 1 without hero" in g.stdout


def test_grader_blind_mode_no_prelabelled(tmp_path):
    """Blind review writes no model_prelabelled.json (zero proposals).

    Absent OR empty must not drop frames or crash: the exclusion set is empty
    and every graded frame counts. The YOLO arm sees all 8 logo frames, so its
    top-1 is still 1.0 (8/8) and the pooled decision runs exactly as before.
    """
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_grader_dryrun.py"),
         "--root", str(tmp_path), "--fps-per-free", "3"],
        capture_output=True, text=True, check=True)

    # absent model_prelabelled.json = blind mode (zero proposals)
    (tmp_path / "model_prelabelled.json").unlink()
    g = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "owlv2_1p7_grade.py"),
         "--export", str(tmp_path),
         "--yolo-dets", str(tmp_path / "realvideo_v1_yolo.json"),
         "--owlv2-dets", str(tmp_path / "realvideo_v1_owlv2.json")],
        capture_output=True, text=True, check=True)
    out = json.loads(
        (Path(str(tmp_path)) / "realvideo_v1_yolo.grade.json").read_text())
    assert out["excluded_circular_for_yolo"] == 0
    assert out["pass"] is True
    assert out["pooled"]["yolo"]["0.05"]["n_logo"] == 8  # no frame dropped
    assert "reviewed frames (8 logo)" in g.stdout

    # empty model_prelabelled.json
    (tmp_path / "model_prelabelled.json").write_text("[]")
    g2 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "owlv2_1p7_grade.py"),
         "--export", str(tmp_path),
         "--yolo-dets", str(tmp_path / "realvideo_v1_yolo.json"),
         "--owlv2-dets", str(tmp_path / "realvideo_v1_owlv2.json")],
        capture_output=True, text=True, check=True)
    out2 = json.loads(
        (Path(str(tmp_path)) / "realvideo_v1_yolo.grade.json").read_text())
    assert out2["excluded_circular_for_yolo"] == 0
    assert out2["pass"] is True