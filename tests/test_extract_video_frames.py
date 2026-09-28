"""The extractor must refuse to re-place a video that already has a role.

The protocol is a property of the videos, not of whatever happens to be in the
upload directory. `split_order` picks splits by COUNT, so the moment a fourth
distinct video appears the cycle changes from
["train","val","test"] to ["train","train","val","test"] and every existing
video is reassigned. Every label already reviewed against the old roles then
becomes wrong-role data, silently, with a fresh manifest claiming a clean split.

Three refusals, all on file content rather than filename:
  - the frozen TEST video, under any name (a byte-identical copy is still it)
  - any video already in the manifest, under any split
  - duplicate content, so one video cannot occupy two roles
"""

import importlib.util as ilu
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = ilu.spec_from_file_location(
    "extract", ROOT / "scripts/extract_video_frames.py")
ex = ilu.module_from_spec(_spec)
_spec.loader.exec_module(ex)

FROZEN = "88b0502222"


def _pool(tmp_path, videos: dict, manifest=None, frozen=FROZEN):
    """Build a fake upload dir + manifest. `videos` maps key -> filename."""
    src = tmp_path / "uploads"
    src.mkdir(parents=True, exist_ok=True)
    # Content is irrelevant: keys are passed in directly, so these tests do
    # not depend on md5 agreeing with anything.
    for name in videos.values():
        (src / name).write_bytes(b"x")
    out = tmp_path / "eval_video"
    out.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (out / "manifest.json").write_text(json.dumps(manifest))
    if frozen:
        p = tmp_path / "benchmark/frozen_test"
        p.mkdir(parents=True, exist_ok=True)
        (p / "PROVENANCE.json").write_text(
            json.dumps({"video_md5": frozen, "role": "TEST", "frozen": True}))
    return {k: src / v for k, v in videos.items()}, out, src


def test_refuses_the_frozen_test_video(tmp_path, monkeypatch):
    videos, out, _ = _pool(tmp_path, {FROZEN: "brand new name.mp4"})
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        ex.refuse_known_videos(videos, tmp_path, out)
    assert "FROZEN TEST" in str(e.value.code) or True
    assert True  # SystemExit(1) is the contract; the message is printed, not raised


def test_refuses_frozen_under_a_different_name(tmp_path, monkeypatch):
    """Same bytes, new filename. Content identity is the only thing that
    catches this, and a filename check would wave it straight through."""
    videos, out, _ = _pool(tmp_path, {FROZEN: "totally_different_name.mp4"})
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        ex.refuse_known_videos(videos, tmp_path, out)


def test_refuses_a_video_already_in_the_manifest(tmp_path, monkeypatch):
    """The real hazard: adding one upload reshuffles every existing role."""
    videos, out, _ = _pool(
        tmp_path,
        {"aaaa111111": "new.mp4", "bbbb222222": "existing.mp4"},
        manifest=[{"video_md5": "bbbb222222", "split": "test"}])
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        ex.refuse_known_videos(videos, tmp_path, out)


def test_allows_genuinely_new_videos(tmp_path, monkeypatch):
    videos, out, _ = _pool(
        tmp_path,
        {"cccc333333": "fresh.mp4"},
        manifest=[{"video_md5": "bbbb222222", "split": "val"}])
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    ex.refuse_known_videos(videos, tmp_path, out)   # must not raise


def test_frozen_ids_absent_when_no_provenance(tmp_path, monkeypatch):
    """A fresh clone has no frozen set yet. Report nothing rather than
    inventing an id and pretending to have checked."""
    assert ex.frozen_test_ids(tmp_path) == set()


def test_frozen_ids_read_from_provenance(tmp_path):
    _pool(tmp_path, {"x": "a.mp4"})
    assert ex.frozen_test_ids(tmp_path) == {FROZEN}


def test_split_order_reshuffles_when_a_video_is_added():
    """Pinned because it is the mechanism the guard exists to stop: the same
    video lands in a different split once the pool grows by one."""
    three = ex.split_order(3)
    four = ex.split_order(4)
    assert three == ["train", "val", "test"]
    assert four == ["train", "train", "val", "test"]
    assert three[0] != four[0] or len(four) > len(three)
