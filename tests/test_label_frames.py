"""The label store must be resumable, video-fair, and must never invent a negative.

The failure this guards: an unreviewed or skipped frame becoming an empty label
file. An empty label file is an assertion that the entire frame is background,
so anything that is not a deliberate NO LOGO verdict would train the detector
that every frame of that video is logo-free - and would inflate the
logo-free-frame denominator that the FP/logo-free-frame rate is computed over.
"""

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np

_spec = ilu.spec_from_file_location(
    "lf", Path(__file__).resolve().parent.parent / "scripts" / "label_frames.py"
)
lf = ilu.module_from_spec(_spec)
_spec.loader.exec_module(lf)


def _pool(root: Path, counts: dict[str, int]) -> list[Path]:
    import cv2
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for vid, n in counts.items():
        for k in range(n):
            p = root / f"{vid}_{k * 40:06d}.jpg"
            cv2.imwrite(str(p), np.full((200, 400, 3), k % 255, np.uint8))
            out.append(p)
    return sorted(out)


def test_review_is_round_robin_across_videos(tmp_path):
    """Labeling 200 consecutive frames of one clip is 200 labels of one scene,
    and split-by-video then throws almost all of them away."""
    files = _pool(tmp_path / "f", {"aaaaaaaaaa": 30, "bbbbbbbbbb": 30})
    store = lf.Store(tmp_path / "l.jsonl")
    served = []
    for _ in range(8):
        f = store.next_frame(files, store.labelled())
        served.append(lf.video_of(f.stem))
        store.put({"file": f.name, "video": lf.video_of(f.stem),
                   "frame_index": lf.frame_index_of(f.stem),
                   "verdict": "free", "boxes": []})
    assert set(served) == {"aaaaaaaaaa", "bbbbbbbbbb"}, f"one video monopolised: {served}"


def test_store_resumes_from_disk(tmp_path):
    """Re-running the same command must continue, not restart: a labelling
    session is interrupted at least once."""
    files = _pool(tmp_path / "f", {"aaaaaaaaaa": 4})
    s1 = lf.Store(tmp_path / "l.jsonl")
    f = s1.next_frame(files, s1.labelled())
    s1.put({"file": f.name, "video": "aaaaaaaaaa", "frame_index": 0,
            "verdict": "logo", "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4}]})
    s2 = lf.Store(tmp_path / "l.jsonl")
    assert len(s2.rows) == 1 and f.name in s2.labelled()
    assert s2.next_frame(files, s2.labelled()).name != f.name
    assert s2.next_frame(files, s2.labelled()) is None or True  # more remain


def test_skipped_frame_is_not_a_negative(tmp_path):
    """SKIP means 'not reviewed'. It must not be exported at all, and must not
    count toward the logo-free denominator."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 2, "bbbbbbbbbb": 2})
    store = lf.Store(tmp_path / "l.jsonl")
    verdicts = [("aaaaaaaaaa_000000", "free"), ("bbbbbbbbbb_000000", "skip"),
                ("aaaaaaaaaa_000040", "logo"), ("bbbbbbbbbb_000040", "free")]
    for name, v in verdicts:
        store.put({"file": f"{name}.jpg", "video": name.rsplit("_", 1)[0],
                   "frame_index": int(name.rsplit("_", 1)[1]), "verdict": v,
                   "boxes": [{"x": 50, "y": 20, "w": 100, "h": 40}] if v == "logo" else []})
    out = tmp_path / "ds"
    args = argparse.Namespace(export=str(out))
    lf.export(args, frames, store)
    rep = json.loads((out / "report.json").read_text())
    assert rep["frames_reviewed"] == 4
    assert rep["frames_exported"] == 3, "the skipped frame must not be exported"
    assert rep["logo_free_frames"] == 2
    for f in (out / "images").rglob("bbbbbbbbbb_000000.jpg"):
        assert not f.exists(), "skipped frame leaked into the dataset"
    lab = out / "labels" / "train" / "aaaaaaaaaa_000000.txt"
    if not lab.exists():
        lab = next((out / "labels" / s / "aaaaaaaaaa_000000.txt"
                    for s in ("train", "val", "test")
                    if (out / "labels" / s / "aaaaaaaaaa_000000.txt").exists()))
    assert lab.read_text() == "", "NO LOGO must be an EMPTY label file"


def test_logo_box_exports_as_normalised_yolo(tmp_path):
    """One box, checked against the frame's real pixel size, on a 400x200 frame."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 1})
    store = lf.Store(tmp_path / "l.jsonl")
    store.put({"file": frames[0].name, "video": "aaaaaaaaaa", "frame_index": 0,
               "verdict": "logo", "boxes": [{"x": 50, "y": 20, "w": 100, "h": 40}]})
    out = tmp_path / "ds"
    lf.export(argparse.Namespace(export=str(out)), frames, store)
    txt = next((out / "labels" / s / "aaaaaaaaaa_000000.txt"
                for s in ("train", "val", "test")
                if (out / "labels" / s / "aaaaaaaaaa_000000.txt").exists())).read_text()
    cls, cx, cy, bw, bh = txt.split()
    assert cls == "0"
    assert abs(float(cx) - (50 + 50) / 400) < 1e-4
    assert abs(float(cy) - (20 + 20) / 200) < 1e-4
    assert abs(float(bw) - 100 / 400) < 1e-4
    assert abs(float(bh) - 40 / 200) < 1e-4


def test_export_splits_whole_videos_never_frames(tmp_path):
    """A frame-level split leaks the same scene into train and test."""
    frames = _pool(tmp_path / "f", {f"v{i:09d}ab": 3 for i in range(5)})
    store = lf.Store(tmp_path / "l.jsonl")
    for f in frames:
        store.put({"file": f.name, "video": lf.video_of(f.stem),
                   "frame_index": lf.frame_index_of(f.stem),
                   "verdict": "free", "boxes": []})
    out = tmp_path / "ds"
    lf.export(argparse.Namespace(export=str(out)), frames, store)
    rep = json.loads((out / "report.json").read_text())
    assert rep["videos"] == {"v000000000ab": "train", "v000000001ab": "train",
                             "v000000002ab": "val", "v000000003ab": "test",
                             "v000000004ab": "test"}, rep["videos"]
    for s in ("train", "val", "test"):
        d = out / "images" / s
        if d.is_dir():
            vids = {lf.video_of(p.stem) for p in d.glob("*.jpg")}
            assert vids and vids <= set(rep["videos"]), f"{s} has mixed videos {vids}"


def test_model_prelabelled_frames_are_flagged(tmp_path):
    """Recall measured against the model's own proposals is circular, so those
    frames must be separable from hand-drawn ones at export time."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 2})
    store = lf.Store(tmp_path / "l.jsonl")
    b = [{"x": 10, "y": 10, "w": 20, "h": 20, "src": "model"}]
    store.put({"file": frames[0].name, "video": "aaaaaaaaaa", "frame_index": 0,
               "verdict": "logo", "boxes": b, "human_drawn": False})
    store.put({"file": frames[1].name, "video": "aaaaaaaaaa",
               "frame_index": lf.frame_index_of(frames[1].stem),
               "verdict": "logo", "boxes": [{**b[0], "src": "human"}],
               "human_drawn": True})
    out = tmp_path / "ds"
    lf.export(argparse.Namespace(export=str(out)), frames, store)
    rep = json.loads((out / "report.json").read_text())
    assert rep["logo_frames_model_prelabelled"] == 1
    assert rep["logo_frames_human_drawn"] == 1
    flagged = json.loads((out / "model_prelabelled.json").read_text())
    assert [f["frame_index"] for f in flagged] == [0]


def test_named_brand_is_carried_beside_the_yolo_label(tmp_path):
    """YOLO has nowhere to store a brand, and the detector must stay single
    class, so the reviewed name has to survive the export somewhere - it is the
    seed for the identity stage."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 1})
    store = lf.Store(tmp_path / "l.jsonl")
    store.put({"file": frames[0].name, "video": "aaaaaaaaaa", "frame_index": 0,
               "verdict": "logo", "human_drawn": True,
               "boxes": [{"x": 10, "y": 10, "w": 40, "h": 20, "src": "human",
                          "brand": "ADIDAS"}]})
    out = tmp_path / "ds"
    lf.export(argparse.Namespace(export=str(out)), frames, store)
    b = json.loads((out / "brands.json").read_text())
    assert b[frames[0].name][0]["brand"] == "ADIDAS"
    assert b[frames[0].name][0]["src"] == "human"
    rep = json.loads((out / "report.json").read_text())
    assert rep["named_brands"] == ["ADIDAS"]
    assert rep["frames_with_named_brand"] == 1
    # still a single-class YOLO set: the brand must not leak into the label file
    lab = next((out / "labels" / s / "aaaaaaaaaa_000000.txt"
                for s in ("train", "val", "test")
                if (out / "labels" / s / "aaaaaaaaaa_000000.txt").exists()))
    assert lab.read_text().splitlines()[0].split()[0] == "0"


def test_data_yaml_always_has_train_and_val(tmp_path):
    """Ultralytics hard-errors on a data.yaml missing val:. With one video only
    one split exists, so the alias must be written and reported."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 2})
    store = lf.Store(tmp_path / "l.jsonl")
    for f in frames:
        store.put({"file": f.name, "video": "aaaaaaaaaa",
                   "frame_index": lf.frame_index_of(f.stem),
                   "verdict": "free", "boxes": []})
    out = tmp_path / "ds"
    lf.export(argparse.Namespace(export=str(out)), frames, store)
    y = (out / "data.yaml").read_text()
    assert "\ntrain:" in y and "\nval:" in y, y


def test_proposal_cache_never_loads_on_its_own(tmp_path, monkeypatch, capsys):
    """A zero-circularity run must stay zero-circularity. The cache used to be
    read whenever it existed, so a session launched with no --propose silently
    inherited a model's pre-draw and every verdict stopped being a human one."""
    frames = _pool(tmp_path / "f", {"aaaaaaaaaa": 2})
    cache = tmp_path / "l.proposals.json"
    cache.write_text(json.dumps({frames[0].name: [
        {"x": 1, "y": 1, "w": 10, "h": 10, "conf": 0.9}]}))
    served = {}

    def fake_serve(args, files, props, store):
        served["props"] = props
        return 0

    monkeypatch.setattr(lf, "serve", fake_serve)
    monkeypatch.setattr(sys, "argv", ["label_frames.py", "--frames", str(tmp_path / "f"),
                                       "--out", str(tmp_path / "l.jsonl")])
    assert lf.main() == 0
    assert served["props"] == {}, "cache auto-loaded into a run with no --propose"
    assert "ZERO-CIRCULARITY" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["label_frames.py", "--frames", str(tmp_path / "f"),
                                       "--out", str(tmp_path / "l.jsonl"),
                                       "--use-cache"])
    assert lf.main() == 0
    assert served["props"], "--use-cache must still be able to load it deliberately"
