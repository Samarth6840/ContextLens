"""A review sheet must be reviewable: spread across videos, traceable, and its
index list must round-trip to files.

A sheet of 48 tiles all cut from ONE video looks like 48 samples but is one
scene's worth of evidence, and nothing in the tile said which shot it was, so a
returned index could not be checked or resolved to a frame. These lock in the
three properties that fix that.
"""

import importlib.util as ilu
from pathlib import Path

import numpy as np

_spec = ilu.spec_from_file_location(
    "vls", Path(__file__).resolve().parent.parent / "scripts" / "video_label_sheet.py"
)
vls = ilu.module_from_spec(_spec)
_spec.loader.exec_module(vls)

_evf = ilu.spec_from_file_location(
    "evf", Path(__file__).resolve().parent.parent / "scripts" / "extract_video_frames.py"
)
evf = ilu.module_from_spec(_evf)
_evf.loader.exec_module(evf)


def test_three_videos_give_a_real_three_way_holdout():
    """2 videos can only ever be a 2-way holdout; the 3rd is what unlocks
    train/val/test, and every video must land in exactly one split."""
    assert evf.split_order(1) == ["train", "test"]
    assert evf.split_order(2) == ["train", "test"]
    assert evf.split_order(3) == ["train", "val", "test"]
    for n in (3, 4, 5, 7):
        order = evf.split_order(n)
        got = [order[i % len(order)] for i in range(n)]
        assert set(got) == {"train", "val", "test"}, f"{n} videos -> {got}"


def test_frame_cap_spreads_over_the_timeline():
    """A cap must not take a prefix: the last frame of the clip has to survive
    or the pool is biased to the opening scene."""
    got = evf.spread(100, 8)
    assert len(got) == 8
    assert got[0] == 0 and got[-1] == 99
    assert got == sorted(set(got))
    assert evf.spread(5, 20) == [0, 1, 2, 3, 4]
    assert evf.spread(100, 1) == [0]


def _pool(root: Path, counts: dict[str, int]) -> list[Path]:
    """counts: {video_md5: n_frames} -> flat dir of frame files, as
    benchmark/eval_video/images/<split> actually is."""
    import cv2
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for vid, n in counts.items():
        for k in range(n):
            p = root / f"{vid}_{k * 40:06d}.jpg"
            cv2.imwrite(str(p), np.full((64, 64, 3), k % 255, np.uint8))
            out.append(p)
    return out


def test_sheet_is_not_one_video(tmp_path):
    """One video with 200 frames must not fill a 48-tile sheet on its own."""
    frames = _pool(tmp_path, {"viddd01aaa": 200})
    picked = vls.pick_frames(frames, sheet=48, per_video=16)
    assert len(picked) == 16, "per-video cap must hold when there is only one video"
    assert len({vls.video_of(p.stem) for p in picked}) == 1


def test_sheet_spreads_across_videos(tmp_path):
    """Three videos, 48 tiles, 16 each - never 48 from whichever sorts first."""
    frames = _pool(tmp_path, {"aaa0000001": 100, "bbb0000002": 100, "ccc0000003": 100})
    picked = vls.pick_frames(frames, sheet=48, per_video=16)
    mix = {v: sum(1 for p in picked if vls.video_of(p.stem) == v)
           for v in ("aaa0000001", "bbb0000002", "ccc0000003")}
    assert len(picked) == 48
    assert all(n == 16 for n in mix.values()), f"uneven video mix: {mix}"


def test_picks_span_the_video_not_a_prefix(tmp_path):
    """The cap must not bias the pool to the opening scene: first and last
    frames of the clip have to be represented."""
    frames = _pool(tmp_path, {"aaa0000001": 100})
    picked = vls.pick_frames(frames, sheet=0, per_video=8)
    idx = sorted(vls.frame_index_of(p.stem) for p in picked)
    assert len(idx) == 8
    assert idx[0] == 0 and idx[-1] == 99 * 40, f"cap took a prefix, not a spread: {idx}"


def test_apply_round_trips_indices_to_files(tmp_path):
    """A returned index list must resolve to exactly the frames it names, and
    land as a YOLO negative (empty label)."""
    frames = _pool(tmp_path, {"aaa0000001": 40, "bbb0000002": 40})
    picked = vls.pick_frames(frames, sheet=16, per_video=8)
    index = [{"i": i, "file": p.name, "video": vls.video_of(p.stem),
              "frame_index": vls.frame_index_of(p.stem)}
             for i, p in enumerate(picked)]
    out = tmp_path / "labelled"
    vls.apply_review(index, {0, 5, 9}, out, tmp_path)

    imgs = sorted(p.name for p in (out / "images").glob("*.jpg"))
    assert len(imgs) == 3
    assert all((out / "labels" / f"{n[:-4]}.txt").read_text() == "" for n in imgs), \
        "a reviewed negative must be an EMPTY label file (YOLO background)"
    for e, n in zip([index[0], index[5], index[9]], sorted(imgs)):
        assert f"{e['video']}_{e['frame_index']:06d}.jpg" == n
    assert "nc: 1" in (out / "data.yaml").read_text()


def test_apply_rejects_index_not_on_sheet(tmp_path):
    """A typo'd index must fail loudly, not silently drop the frame."""
    frames = _pool(tmp_path, {"aaa0000001": 10})
    picked = vls.pick_frames(frames, sheet=8, per_video=8)
    index = [{"i": i, "file": p.name, "video": vls.video_of(p.stem),
              "frame_index": vls.frame_index_of(p.stem)}
             for i, p in enumerate(picked)]
    import pytest
    with pytest.raises(SystemExit):
        vls.apply_review(index, {99}, tmp_path / "x", tmp_path)
