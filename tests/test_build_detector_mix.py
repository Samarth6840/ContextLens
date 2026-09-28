"""The negative cap must fall on the crops and spare the human negatives.

The mix shipped 390 mined crops + 64 human logo-free frames against 124
positive boxes (3.66:1). The fine-tune that read it ended below its own
pretrained init - val mAP50 0.0015 -> 0.00006 by epoch 5 - because at that
ratio "emit nothing" is the cheapest prediction available.

So the cap is tested where it is load-bearing: crops get dropped, and the
reviewed full-frame negatives survive even when they alone exceed the budget,
because they are the human ground truth and the crops are not.
"""

import importlib.util as ilu
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = ilu.spec_from_file_location(
    "mix", ROOT / "scripts" / "build_detector_mix.py")
build = ilu.module_from_spec(_spec)
_spec.loader.exec_module(build)

TRAIN_V = "aaaa111111"
VAL_V = "bbbb222222"


def _frame(path, boxes=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((200, 200, 3), 255, np.uint8))
    lines = [f"0 {(x + w/2)/200:.6f} {(y + w/2)/200:.6f} {w/200:.6f} {w/200:.6f}"
             for x, y, w in boxes]
    (path.parent.parent / "labels").mkdir(parents=True, exist_ok=True)
    (path.parent.parent / "labels" / f"{path.stem}.txt").write_text(
        "".join(l + "\n" for l in lines))


def _dataset(tmp_path, pos_frames, neg_frames, crops):
    frames = tmp_path / "eval_video/images"
    rows = []
    for i in range(pos_frames):                      # 2 boxes each
        f = f"{TRAIN_V}_{i:06d}.jpg"
        _frame(frames / f, [(10, 10, 20), (100, 100, 30)])
        rows.append({"file": f, "video": TRAIN_V, "frame_index": i,
                     "verdict": "logo", "human_drawn": True, "boxes": [
                         {"x": 10, "y": 10, "w": 20, "h": 20},
                         {"x": 100, "y": 100, "w": 30, "h": 30}]})
    for i in range(neg_frames):                      # human logo-free frames
        f = f"{TRAIN_V}_n{i:06d}.jpg"
        _frame(frames / f)
        rows.append({"file": f, "video": TRAIN_V, "frame_index": 900 + i,
                     "verdict": "free", "human_drawn": False, "boxes": []})
    f = f"{VAL_V}_000000.jpg"                        # VAL needs one frame
    _frame(frames / f, [(10, 10, 20)])
    rows.append({"file": f, "video": VAL_V, "frame_index": 0, "verdict": "logo",
                 "human_drawn": True,
                 "boxes": [{"x": 10, "y": 10, "w": 20, "h": 20}]})
    (tmp_path / "eval_video").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval_video/labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    for i in range(crops):                           # mined crops, empty labels
        c = tmp_path / "hard_negatives/images" / f"{TRAIN_V}_000000_{i:04d}.jpg"
        c.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(c), np.full((64, 64, 3), 255, np.uint8))
        (tmp_path / "hard_negatives/labels").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"hard_negatives/labels/{TRAIN_V}_000000_{i:04d}.txt").write_text("")
    return tmp_path


def _run(tmp_path, *extra):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_detector_mix.py"),
         "--labels", str(tmp_path / "eval_video/labels.jsonl"),
         "--frames", str(tmp_path / "eval_video/images"),
         "--hard-negatives", str(tmp_path / "hard_negatives"),
         "--out", str(tmp_path / "mix"),
         "--train-videos", TRAIN_V, "--val-video", VAL_V, *extra],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((tmp_path / "mix/manifest.json").read_text())


def test_crops_are_capped_and_human_negatives_survive(tmp_path):
    """4 boxes (2 pos frames x 2), 3 human negative frames, 20 crops.
    At 1.5:1 the budget is 6 - 3 = 3 crops, so 17 are dropped and every
    human frame is still in train/."""
    tmp_path = _dataset(tmp_path, pos_frames=2, neg_frames=3, crops=20)
    m = _run(tmp_path, "--neg-per-positive", "1.5")
    train = tmp_path / "mix/images/train"
    assert m["train"]["positive_boxes"] == 4
    assert m["hard_negative_crops_dropped"] == 17
    assert m["train"]["hard_negative_crops"] == 3
    assert m["ratio_negative_to_positive"] == 1.5
    for i in range(3):                               # human negatives kept
        assert (train / f"{TRAIN_V}_n{i:06d}.jpg").exists()
    assert len(list(train.glob(f"{TRAIN_V}_*_*.jpg"))) == 3


def test_human_negatives_are_never_dropped_even_when_over_budget(tmp_path):
    """10 human negative frames against 2 boxes is 5:1 on its own. The cap
    cannot buy that back - there are no crops left to cut - so the ratio is
    reported above the cap rather than silently deleting reviewed ground
    truth to make a number look tidy."""
    tmp_path = _dataset(tmp_path, pos_frames=1, neg_frames=10, crops=3)
    m = _run(tmp_path, "--neg-per-positive", "1.0")
    train = tmp_path / "mix/images/train"
    assert m["train"]["hard_negative_crops"] == 0
    assert m["hard_negative_crops_dropped"] == 3
    assert len(list(train.glob(f"{TRAIN_V}_n*.jpg"))) == 10
    assert m["ratio_negative_to_positive"] > 1.0


def test_single_source_positives_are_flagged(tmp_path):
    """The cap cannot manufacture a second video's logos. Say so in the
    manifest instead of letting a report quote the ratio and imply
    generalizable supervision."""
    tmp_path = _dataset(tmp_path, pos_frames=2, neg_frames=1, crops=5)
    m = _run(tmp_path, "--neg-per-positive", "1.5")
    assert m["train"]["videos_carrying_positives"] == [TRAIN_V]
    assert m["train"]["positives_from_one_video"] is True


def test_size_bucket_edges_unchanged():
    """Reused by the manifest's box_size tally; a silent change here would
    re-label every box in every report."""
    assert build.size_bucket(10, 10, 1_000_000) == "tiny"
    assert build.size_bucket(100, 100, 1_000_000) == "small"
    assert build.size_bucket(300, 300, 1_000_000) == "medium"
    assert build.size_bucket(600, 600, 1_000_000) == "large"
