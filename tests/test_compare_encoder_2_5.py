import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TRAIN_MD5 = "1111111111"


def _make_export(root: Path, md5: str, n_logo=1, n_free=2, clobber_hero=None):
    """YOLO-label export: n_logo frames with a hero GT box + n_free logo-free."""
    W, H = 200, 100
    img = np.full((H, W, 3), 128, np.uint8)
    hero = clobber_hero or (40, 20, 120, 60)  # x1,y1,x2,y2
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    for i in range(n_logo):
        stem = f"{md5}_{i:06d}"
        cv2.imwrite(str(root / "images" / "train" / f"{stem}.jpg"), img)
        cx, cy = (hero[0] + hero[2]) / 2 / W, (hero[1] + hero[3]) / 2 / H
        bw, bh = (hero[2] - hero[0]) / W, (hero[3] - hero[1]) / H
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    for i in range(n_logo, n_logo + n_free):
        stem = f"{md5}_{i:06d}"
        cv2.imwrite(str(root / "images" / "train" / f"{stem}.jpg"), img)
        (root / "labels" / "train" / f"{stem}.txt").write_text("")
    return hero


def test_holdout_refusal(tmp_path):
    _make_export(tmp_path, "88b0502222")  # a real holdout md5 prefix
    dets = {}  # fp path never reached: the guard must fire first
    (tmp_path / "realvideo_v1_yolo.json").write_text(json.dumps(dets))
    r = subprocess.run(
        [sys.executable, "scripts/compare_encoder_2_5.py",
         "--export", str(tmp_path),
         "--yolo-dets", str(tmp_path / "realvideo_v1_yolo.json")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "refusing to read holdout" in r.stderr


def test_fp_per_free_frame_train_only(tmp_path, monkeypatch):
    from scripts.compare_encoder_2_5 import fp_per_free_frame, per_train_video
    from scripts.eval_detector_separation import read_split
    hero = _make_export(tmp_path, TRAIN_MD5, n_logo=1, n_free=2)
    rs = read_split(tmp_path, "all")
    free_rows = [r for r in rs if not r[2]]
    assert len(free_rows) == 2

    # one true-ish det on free frame 0 (inside GT? no GT box on a free frame),
    # so ANY det there is a frame FP; 2 dets on free frame 1 -> 3 FPs / 2 free.
    dets = {
        f"{TRAIN_MD5}_000001": [list(hero) + [0.5]],
        f"{TRAIN_MD5}_000002": [list(hero) + [0.5], (5, 5, 8, 8, 0.5)],
    }
    assert fp_per_free_frame(dets, free_rows) == pytest.approx(3 / 2)

    rows = per_train_video(str(tmp_path), dets, run_before=None, run_after=None)
    assert set(rows) == {TRAIN_MD5}
    assert rows[TRAIN_MD5]["n_frames"] == 3
    assert rows[TRAIN_MD5]["n_logo_free"] == 2
    assert rows[TRAIN_MD5]["fp_per_free_frame"] == pytest.approx(1.5)
    assert rows[TRAIN_MD5]["brand_resolutions_before"] == -1  # no run given