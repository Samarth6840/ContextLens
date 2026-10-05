"""The quarantine must be surgical on clean data and total on nested data.

Two failure modes, both tested here:

  * over-aggressive  - a rule that eats legitimate boxes (two logos side by side
    on a shirt, a WebLogo product shot that fills the frame) silently deletes the
    supervision the detector needs.
  * under-aggressive - keeping a box that is 95% inside another box leaves YOLO
    with two mutually contradictory targets, and keeping a frame-sized box as a
    "logo" teaches the model to fire on scenes.
"""

import importlib.util as ilu
from pathlib import Path

import numpy as np
import pytest

_spec = ilu.spec_from_file_location(
    "san", Path(__file__).resolve().parent.parent / "scripts" / "sanitize_dataset.py"
)
san = ilu.module_from_spec(_spec)
_spec.loader.exec_module(san)


def _yolo(boxes):
    """(N,4) xyxy pixels on a 1000x1000 frame -> YOLO label lines."""
    return [f"0 {(x1+x2)/2/1000:.6f} {(y1+y2)/2/1000:.6f} "
            f"{(x2-x1)/1000:.6f} {(y2-y1)/1000:.6f}" for x1, y1, x2, y2 in boxes]


def _root(tmp_path, name, lines):
    (tmp_path / "labels" / name).mkdir(parents=True)
    (tmp_path / "images" / name).mkdir(parents=True)
    (tmp_path / "labels" / name / f"{name}.txt").write_text(
        "".join(l + "\n" for l in lines))
    (tmp_path / "images" / name / f"{name}.jpg").write_bytes(b"not-a-real-jpeg")


def test_nested_scene_yields_nothing_not_a_truncated_guess():
    """A mark inside a frame-sized sign: the sign is not a logo, and the mark
    cannot be kept either because it is 100% inside the sign - two boxes at
    IoU ~1.0 are contradictory assignment targets. YOLO has no nested output,
    so a nested annotation is unrepresentable, not recoverable."""
    b = np.array([[0.50, 0.50, 0.90, 0.90],    # sign, 81% of the frame
                  [0.48, 0.48, 0.56, 0.56]])  # mark, 100% inside the sign
    keep, _ = san.classify(b, max_area=0.50, max_aspect=20.0, leaf=0.95)
    assert not keep.any(), "nested boxes are contradictory, keep neither"


def test_loose_mark_outside_any_container_survives():
    """A real logo is small and does not sit inside another annotation. This is
    the shape the rules must NOT throw away."""
    b = np.array([[0.05, 0.05, 0.95, 0.60],    # banner, overlapping nothing
                  [0.10, 0.80, 0.30, 0.92]])  # mark below it, clear of it
    keep, _ = san.classify(b, max_area=0.50, max_aspect=20.0, leaf=0.95)
    assert keep.tolist() == [False, True], "banner over max_area dies, mark lives"


def test_side_by_side_logos_both_survive():
    """Overlapping is not the same as contained. Two marks on a shirt share
    pixels but neither is inside the other - both are real supervision."""
    b = np.array([[0.30, 0.50, 0.45, 0.60],
                  [0.42, 0.50, 0.55, 0.60]])  # 20% horizontal overlap
    keep, _ = san.classify(b, max_area=0.50, max_aspect=20.0, leaf=0.95)
    assert keep.all()


def test_dense_image_dropped_whole_never_truncated(tmp_path):
    """Every unlabelled region trains as background, so a capped box list is
    worse than no image. The whole image goes to suspect_boxes/."""
    _root(tmp_path, "train", _yolo([[i, i, i + 20, i + 20] for i in range(0, 180, 3)]))
    out = tmp_path / "clean"
    st = san.sanitize_split(tmp_path, out, "train", 0.50, 20.0, 0.95, max_boxes=50)
    assert st["images_out"] == 0 and st["images_dropped_dense"] == 1
    assert not list((out / "labels" / "train").glob("*.txt"))
    assert list((out / "suspect_boxes" / "labels" / "train").glob("*.txt"))


def test_fully_quarantined_image_is_not_written_as_a_negative(tmp_path):
    """A label file with no boxes is an implicit negative: it tells the model
    the whole frame is background. An image that loses every box must leave
    the training set entirely."""
    _root(tmp_path, "train", _yolo([[0, 0, 1000, 1000]]))  # 100% of the frame
    out = tmp_path / "clean"
    st = san.sanitize_split(tmp_path, out, "train", 0.50, 20.0, 0.95, max_boxes=0)
    assert st["images_out"] == 0 and st["images_dropped_all_quarantined"] == 1
    assert not list((out / "labels" / "train").glob("*.txt"))
    assert list((out / "suspect_boxes" / "images" / "train").glob("*.jpg"))


def test_source_root_is_never_modified(tmp_path):
    _root(tmp_path, "train", _yolo([[100, 100, 200, 200], [0, 0, 1000, 1000]]))
    before = (tmp_path / "labels" / "train" / "train.txt").read_text()
    san.sanitize_split(tmp_path, tmp_path / "clean", "train", 0.50, 20.0, 0.95, 0)
    assert (tmp_path / "labels" / "train" / "train.txt").read_text() == before
