"""Quarantine suspect logo boxes instead of deleting them, and say what is left.

Why this exists
---------------
scripts/domain_mismatch.py reports that 8.7% of LogoDet-3K train boxes cover more
than half the frame. Quarantining those is the obvious move and it is the wrong
one on its own, because 8.7% is a symptom of a much larger structural fault.

Measured on weights/logo_detector/labels/train (101,873 boxes, 201 images with
annotations, 2026-09-28):

    containment >0.95  ->  keep 499 boxes   (0.5%)
    containment >0.90  ->  keep 321 boxes   (0.3%)
    control: OpenLogo   ->  keep 99.8% of boxes

99.5% of LogoDet boxes lie ~entirely inside another box of the same image. In
train/1.txt, all 311 boxes larger than half the frame overlap every other one
(median pairwise IoU 0.68) and their union covers 100% of the image. That is not
scattered label noise, it is a nested containment hierarchy flattened into a
flat box list: text blocks containing text lines containing marks. YOLO cannot
represent nested boxes at all - two boxes with IoU 0.95 are mutually
contradictory assignment targets - so the model is handed an unsatisfiable
objective and the best it can do is emit the container scale.

Which is exactly what it emits. Box geometry, normalised by frame:

                       boxes    w/frame p50   area p50   w>0.5   area<1%
    LogoDet train      101873        0.487       0.116   48.6%     3.5%
    OpenLogo train      31659        0.107       0.0095   6.7%    51.2%
    ContextLens video     9053       0.587       0.213   75.6%     0.0%
    (video row = detector predictions, not reviewed truth)

The detector on real video frames reproduces the LogoDet distribution and is
indistinguishable from it. tiny recall is 0 because LogoDet contains almost no
sub-1%-of-frame supervision, not because the labels are 8.7% dirty.

So the quarantine has three rules, applied per box:

  1. containment  - a box ~entirely inside another box is a container, not a
                    mark. Only leaves survive. This is the rule that matters;
                    the area rule alone leaves the objective broken.
  2. area         - box > --max-area of the frame is a scene / advertisement /
                    card / full-screen region, not a logo.
  3. aspect       - a logo mark is not a 40:1 sliver.

and one rule per image, unchanged from export_yolo: an image over
--max-boxes-per-image is dropped WHOLE, never truncated, because every
unlabelled region trains as background. An image whose boxes are ALL quarantined
is dropped whole for the same reason - it must not become an implicit negative.

Nothing is deleted. Quarantined boxes are written to suspect_boxes/ and the
source root is never modified, so a rule can be relaxed by re-running.

    python scripts/sanitize_dataset.py --root weights/logo_detector \
        --out weights/logo_detector_clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def to_xyxy(b: np.ndarray) -> np.ndarray:
    """(N,4) YOLO cx,cy,w,h -> (N,4) x1,y1,x2,y2."""
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def containment_max(b: np.ndarray) -> np.ndarray:
    """For each box, the largest FRACTION of it that lies inside any one other box.

    This is the containment test, deliberately not IoU. A mark inside a sign is
    0.9 IoU with the sign and must die; two logos side by side on a shirt share
    a small IoU but neither contains the other and both must live. Fraction
    covered answers that question and IoU does not.
    """
    if len(b) < 2:
        return np.zeros(len(b))
    x1, y1, x2, y2 = to_xyxy(b).T
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    iw = np.clip(np.minimum(x2[:, None], x2[None])
                 - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None])
                 - np.maximum(y1[:, None], y1[None]), 0, None)
    cov = (iw * ih) / np.maximum(area[:, None], 1e-9)
    np.fill_diagonal(cov, 0.0)
    return cov.max(1)


def classify(boxes: np.ndarray, max_area: float, max_aspect: float,
             leaf: float) -> tuple[np.ndarray, Counter]:
    """-> (keep mask, per-reason quarantine counts). Reasons are ordered, so a
    box is attributed to the first rule that caught it and the counts sum to the
    number of quarantined boxes."""
    reason = Counter()
    cov = containment_max(boxes)
    area = boxes[:, 2] * boxes[:, 3]
    aspect = boxes[:, 2] / np.maximum(boxes[:, 3], 1e-9)
    bad = np.zeros(len(boxes), bool)
    for name, m in (("contained", cov > leaf),
                    ("over_max_area", area > max_area),
                    ("extreme_aspect", np.maximum(aspect, 1 / np.maximum(aspect, 1e-9)) > max_aspect)):
        m = m & ~bad
        reason[name] = int(m.sum())
        bad |= m
    return ~bad, reason


_STATS = ("images_in", "images_out", "images_dropped_dense",
          "images_dropped_all_quarantined", "images_dropped_no_annotation",
          "boxes_in", "boxes_kept", "boxes_quarantined",
          "boxes_quarantined_dense", "contained", "over_max_area",
          "extreme_aspect")


def sanitize_split(root: Path, out: Path, split: str, max_area: float,
                   max_aspect: float, leaf: float, max_boxes: int) -> dict:
    lab_dir = root / "labels" / split
    img_dir = root / "images" / split
    stat = Counter({k: 0 for k in _STATS})
    for lab in sorted(lab_dir.glob("*.txt")):
        stat["images_in"] += 1
        rows = [l.split() for l in lab.read_text().splitlines() if l.split()]
        raw = [r for r in rows if len(r) >= 5]
        stat["boxes_in"] += len(raw)
        img = img_dir / f"{lab.stem}.jpg"
        if not raw or not img.exists():
            stat["images_dropped_no_annotation"] += 1
            continue
        b = np.array([[float(v) for v in r[1:5]] for r in raw])
        if max_boxes and len(b) > max_boxes:
            # Whole image out, never a truncated box list: the discarded boxes
            # would train as background (see test_export_label_integrity.py).
            stat["images_dropped_dense"] += 1
            stat["boxes_quarantined_dense"] += len(b)
            _quarantine(out, split, lab, img, raw, ["over_density"])
            continue
        keep, reason = classify(b, max_area, max_aspect, leaf)
        stat.update(reason)
        if not keep.any():
            stat["images_dropped_all_quarantined"] += 1
            stat["boxes_quarantined"] += len(b)
            _quarantine(out, split, lab, img, raw, ["all_quarantined"])
            continue
        stat["boxes_kept"] += int(keep.sum())
        stat["images_out"] += 1
        _write(out / "labels" / split / f"{lab.stem}.txt",
               [r for r, k in zip(raw, keep) if k])
        _link(out / "images" / split / img.name, img)
    return dict(stat)


def _write(p: Path, rows: list[list[str]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(" ".join(r) + "\n" for r in rows))


def _link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.hardlink_to(src)
        except OSError:  # different filesystem, or no hardlink support
            shutil.copy2(src, dst)


def _quarantine(out: Path, split: str, lab: Path, img: Path,
                raw: list[list[str]], reasons: list[str]) -> None:
    """Park the labels (and a hardlink to the image) under suspect_boxes/ so a
    human can review or relabel them instead of them being silently destroyed."""
    _write(out / "suspect_boxes" / "labels" / split / lab.name, raw)
    _link(out / "suspect_boxes" / "images" / split / img.name, img)
    (out / "suspect_boxes" / "reasons" / split).mkdir(parents=True, exist_ok=True)
    (out / "suspect_boxes" / "reasons" / split / lab.name).write_text(
        " ".join(reasons) + "\n")


def _carry_negatives(root: Path, out: Path) -> None:
    """Mined negatives are already-empty labels - nothing to sanitise, and they
    are the only clean logo-free supervision in these roots, so carry them."""
    for sub in ("negatives", "labels_neg"):
        src = root / sub
        if src.is_dir():
            shutil.copytree(src, out / sub, dirs_exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-area", type=float, default=0.50,
                    help="quarantine boxes larger than this fraction of the frame")
    ap.add_argument("--max-aspect", type=float, default=20.0,
                    help="quarantine boxes more elongated than this (w:h or h:w)")
    ap.add_argument("--leaf", type=float, default=0.95,
                    help="quarantine a box this fraction inside any other box")
    ap.add_argument("--max-boxes-per-image", type=int, default=0,
                    help="drop whole images over this density (0 = no cap)")
    ap.add_argument("--hard", action="store_true",
                    help="overwrite an existing --out")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if not (root / "labels").is_dir():
        sys.exit(f"FATAL: no labels/ under {root}")
    if out.exists():
        if not args.hard:
            sys.exit(f"FATAL: {out} exists; pass --hard to overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    report = {"root": str(root.resolve()), "out": str(out.resolve()),
              "rules": {"max_area": args.max_area, "max_aspect": args.max_aspect,
                        "leaf_containment": args.leaf,
                        "max_boxes_per_image": args.max_boxes_per_image},
              "splits": {}}
    for s in args.splits:
        if not (root / "labels" / s).is_dir():
            continue
        report["splits"][s] = sanitize_split(
            root, out, s, args.max_area, args.max_aspect, args.leaf,
            args.max_boxes_per_image)
    _carry_negatives(root, out)

    total = Counter()
    for st in report["splits"].values():
        total.update(st)
    report["total"] = dict(total)
    kept, inb = total["boxes_kept"], total["boxes_in"]
    report["kept_fraction"] = round(kept / inb, 4) if inb else None
    report["boxes_per_image_kept"] = (
        round(kept / total["images_out"], 2) if total["images_out"] else 0.0)
    report["boxes_per_image_in"] = (
        round(inb / max(1, total["images_in"]), 2) if total["images_in"] else 0.0)
    # A detector needs thousands of positive boxes. Density landing near the
    # video target is not evidence of health if only a few hundred boxes survive.
    report["verdict"] = (
        "usable" if kept >= 2000 and total["images_out"] >= 50 else
        f"UNUSABLE: {kept} clean boxes over {total['images_out']} images. "
        "Do not retrain on this. Keep the clean source, get real video labels.")
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{'split':<8}{'images in':>11}{'images out':>12}{'boxes in':>11}"
          f"{'boxes kept':>12}{'kept %':>9}{'box/img':>10}")
    for s, st in report["splits"].items():
        n_in, k = st["images_in"], st["boxes_kept"]
        print(f"{s:<8}{st['images_in']:>11}{st['images_out']:>12}"
              f"{st['boxes_in']:>11}{k:>12}{100*k/max(1,st['boxes_in']):>8.1f}%"
              f"{k/max(1,st['images_out']):>10.2f}")
    print(f"\nquarantine reasons: {dict(Counter({k: v for k, v in total.items() if 'quarant' in k or k in ('contained','over_max_area','extreme_aspect')}))}")
    print(f"kept {kept}/{inb} boxes ({100*report['kept_fraction'] or 0:.1f}%), "
          f"{report['boxes_per_image_in']} -> {report['boxes_per_image_kept']} boxes/image")
    print(f"quarantined -> {out}/suspect_boxes/    report -> {out}/report.json")
    print(f"verdict: {report['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
