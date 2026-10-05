"""
Train a single-class LOGO detector on real LogoDet-3K boxes.

Why
---
Measured on 30 real LogoDet-3K frames / 41 logo boxes (scripts/eval_logo.py):

    YOLO-World, 6 generic queries    0 TP   2 FP   F1 0.000
    YOLO-World, 45 queries          3 TP  99 FP   F1 0.042

YOLO-World prompted with text is not a logo detector in either configuration.
The open-vocabulary head has no notion of "logo" — it matches the *prompted*
concept, and a logo it has never been told about is invisible to it. Prompting
it with 39 brand names does not fix that; it just manufactures a false positive
per brand per frame.

The fix is a detector with weights that have actually SEEN logos. LogoDet-3K
provides ~200k human-annotated logo boxes on real photographs, which is exactly
the supervision this module was missing.

Single class on purpose
-----------------------
The model learns "logo" vs "not logo" and nothing else. Brand identity is NOT
its job: that already exists downstream and is fail-closed (crop-OCR, then CLIP
retrieval, then UNKNOWN). A detector that also guesses brands re-creates the
fabrication path this architecture removed. So the training label is one class,
and the eval stays honest about the thing we actually need — finding the box.

    python scripts/train_logo_detector.py \
        --parquet /tmp/realbench/data --out weights/logo_detector
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def split_for(path: str, val_frac: float, test_frac: float) -> str:
    import hashlib

    h = int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def export_yolo(parquet_dir: Path, out: Path, val_frac: float, test_frac: float,
                min_area: float, max_area: float, limit_per_split: int,
                max_boxes_per_image: int = 50, seed: int = 17):
    """Write YOLO-format labels + images, splitting by hashed image path.

    Every LogoDet box becomes class 0 ('logo'). Boxes outside the area bounds are
    dropped; images left with no usable box are skipped rather than shipped as
    implicit negatives, because LogoDet images are logo-dense and calling them
    negative would teach the model that logos are background.

    YOLO trains every UNLABELLED region as background. So the only two safe
    operations on a label file are: keep every box, or drop the whole image.
    Dropping an individual box silently teaches the model that a real,
    human-annotated logo is background. LogoDet-3K annotates exhaustively
    (boxes/image: median 161, mean 511, p90 1651; boxes are distinct, only
    1.1x duplication), and their median area is 11.7% of the frame, so this
    matters: the previous version subsampled dense images to 50 boxes via
    np.linspace, which pushed 13.9x more box-area into the background class
    than into the positive class. That is a precision-destroying label
    corruption, not a convenience trade-off. Hence: no subsampling, and any
    box outside the area bounds disqualifies the entire image.
    """
    import cv2
    import pandas as pd
    import pyarrow.parquet as pq

    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        sys.exit(f"FATAL: no *.parquet under {parquet_dir}")
    names = json.loads(
        pq.ParquetFile(shards[0]).schema_arrow.metadata[b"huggingface"].decode()
    )["info"]["features"]["company_name"]["names"]

    by_image: dict[str, list] = defaultdict(list)
    for shard in shards:
        df = pd.read_parquet(shard, columns=["image_path", "company_name", "bbox"])
        for img, cls, bbox in zip(df["image_path"], df["company_name"], df["bbox"]):
            path = img.get("path") if isinstance(img, dict) else str(img)
            if bbox is None or len(bbox) < 4:
                continue
            by_image[path].append([float(v) for v in list(bbox)[:4]])

    counts = defaultdict(int)
    written = 0
    dropped_dense = 0
    dropped_area = 0
    seen = set()
    for shard in shards:
        df = pd.read_parquet(shard, columns=["image_path"])
        for img in df["image_path"]:
            path = img.get("path") if isinstance(img, dict) else str(img)
            if path not in by_image or path in seen:
                continue
            seen.add(path)
            # Dense images are DROPPED, never subsampled. Keeping the image and
            # capping its box list marks the discarded logos as background.
            boxes = by_image[path]
            if max_boxes_per_image and len(boxes) > max_boxes_per_image:
                dropped_dense += 1
                continue
            split = split_for(path, val_frac, test_frac)
            if limit_per_split and counts[split] >= limit_per_split:
                continue
            raw = img.get("bytes") if isinstance(img, dict) else None
            if not isinstance(raw, (bytes, bytearray)):
                continue
            arr = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            h, w = arr.shape[:2]
            lines = []
            rejected = False
            for x1, y1, x2, y2 in boxes:
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 <= x1 or y2 <= y1:
                    # Entirely outside the frame. Nothing visible to label and
                    # no in-image region turned into background, so skip it.
                    # This is the common case: LogoDet boxes are in original-image
                    # coordinates and many overhang a crop. Clipping above keeps
                    # the visible part of a partially-outside box labelled.
                    continue
                area = (x2 - x1) * (y2 - y1) / float(w * h)
                if area < min_area or area > max_area:
                    rejected = True
                    break
                # YOLO normalised cx, cy, w, h
                lines.append(
                    f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                    f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                )
            if rejected or not lines:
                # A box we cannot write is a logo we would train as background.
                dropped_area += 1
                continue
            d = out / "images" / split
            dl = out / "labels" / split
            d.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            name = f"{Path(path).stem}.jpg"
            cv2.imwrite(str(d / name), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            (dl / f"{name[:-4]}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1
            written += 1
    data = out / "data.yaml"
    out.mkdir(parents=True, exist_ok=True)
    # The negative crops live in negatives/<split>, not images/<split>, because
    # Ultralytics only discovers label files under images/ and would silently
    # ignore mined negatives written there.
    base = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
    negs = {s: len(list((out / "negatives" / s).glob("*.jpg"))) for s in ("train", "val")}
    if any(negs.values()):
        data.write_text(
            f"{base}# mined logo-free negatives (empty label files)\n"
            f"train: [images/train, negatives/train]\n"
            f"val: [images/val, negatives/val]\n\nnc: 1\nnames: ['logo']\n"
        )
        print(f"data.yaml adds {negs['train']} train / {negs['val']} val mined negatives")
    else:
        data.write_text(f"{base}\nnc: 1\nnames: ['logo']\n")
    print(f"exported {written} images: {dict(counts)}")
    print(f"dropped {dropped_dense} images over the density cap "
          f"({max_boxes_per_image}/image) - NOT subsampled")
    print(f"dropped {dropped_area} images with a box outside area "
          f"[{min_area},{max_area}] - would have trained a logo as background")
    print(f"dataset yaml -> {data}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="weights/logo_detector")
    ap.add_argument("--base", default="yolov8s.pt", help="start from COCO yolov8s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.25)
    # YOLO treats every unlabelled region as background, so boxes may never be
    # dropped individually. min-area 0 / max-area 1 / no density cap therefore
    # label every box in all 310 LogoDet images (157,056 boxes). 56% of the
    # eval-set boxes fall inside LogoDet's interquartile size band, so the two
    # sources are scale-compatible.
    ap.add_argument("--min-area", type=float, default=0.0)
    ap.add_argument("--max-area", type=float, default=1.0)
    ap.add_argument("--limit-per-split", type=int, default=0, help="0 = all")
    ap.add_argument("--max-boxes-per-image", type=int, default=0,
                    help="drop exhaustively-annotated images, never subsample (0 = keep all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--export-only", action="store_true",
                    help="write the YOLO dataset and stop (no training)")
    ap.add_argument("--skip-export", action="store_true",
                    help="train on the dataset already in --out (e.g. a corpus "
                         "built by build_openlogo_yolo.py) instead of re-exporting")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the training set to use (Ultralytics)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.skip_export:
        counts = {s: len(list((out / "images" / s).glob("*.jpg")))
                  for s in ("train", "val") if (out / "images" / s).exists()}
        negs = {s: len(list((out / "negatives" / s).glob("*.jpg")))
                for s in ("train", "val") if (out / "negatives" / s).exists()}
        print(f"using existing dataset in {out}: {counts} + {negs} negatives")
    else:
        counts = export_yolo(
            Path(args.parquet), out, args.val_frac, args.test_frac,
            args.min_area, args.max_area, args.limit_per_split,
            max_boxes_per_image=args.max_boxes_per_image,
        )
        if args.export_only:
            return 0
    if counts.get("train", 0) < 50:
        sys.exit(f"FATAL: only {counts.get('train',0)} train images; need >=50. "
                 "Download more LogoDet-3K shards or lower --limit-per-split constraints.")

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=str(out / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out),
        name="train",
        exist_ok=True,
        # Small logos: keep the mosaic OFF so a 60x55 logo is not shrunk to
        # nothing, and keep a low mosaic scale for the rest.
        mosaic=0.5,
        scale=0.5,
        hsv_v=0.3,
        # A logo is a small, high-contrast, flat mark. Heavy augmentation invents
        # textures that do not occur and destroys exactly the cue we need.
        degrees=5.0,
        translate=0.1,
        fliplr=0.5,
        patience=15,
        # LabelDet frames are logo-dense (median 161 boxes, p90 1651). The
        # default max_det=300 would silently cap the training signal, and a
        # cap also throttles recall at inference on dense frames.
        max_det=2000,
        fraction=args.fraction,
    )
    best = out / "train" / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
