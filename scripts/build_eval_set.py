"""
Build a REAL labeled logo-detection eval set from LogoDet-3K parquet shards.

Why this exists
---------------
`fusion_eval/dataset.py` synthesises its eval set by compositing reference-bank
logos onto noise backgrounds. That measures compositing, not detection: every
logo is clean, axis-aligned, fully opaque and at a fixed scale, so the numbers
say nothing about real frames. The only real detection measurement that ever
existed (benchmark_*_20260805_*.json) was deleted in commit 40ec5bf.

This builder produces the real thing from LogoDet-3K (MIT), which ships genuine
photographs with human bounding boxes across 3000 logo classes.

Emits
-----
    <out>/images/<split>/<image_id>.jpg   real frame
    <out>/labels.json    {split: [{image, boxes:[{bbox,brand}], negative}]}
    <out>/summary.json   per-brand counts + split sizes

Honesty constraints (deliberate — do not relax)
-----------------------------------------------
* Only LogoDet classes that map to a catalog brand via
  `src.brand_catalog.match_brand` become positives. No brand is invented.
* Split is a stable hash of the image path, so an image is ALWAYS in the same
  split and no logo can leak across splits.
* NEGATIVES are real images whose logos all fall OUTSIDE the catalog: the
  detector must localise a logo-shaped region and then ABSTAIN rather than name
  a catalog brand. These are the rows that make abstention measurable.
* `min_area` / `max_area` drop boxes too small to crop or so large they are
  editorial straps rather than logos.

Two passes so we never hold 3 GB of image bytes in RAM: pass 1 indexes
paths/classes/boxes only, pass 2 re-reads shards one at a time and writes each
wanted image straight to disk.

Usage
-----
    python scripts/build_eval_set.py \
        --parquet /tmp/realbench/data \
        --out benchmark/eval_real --val-frac 0.15 --test-frac 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLITS = ("train", "val", "test")


def split_for(path: str, val_frac: float, test_frac: float) -> str:
    """Stable per-image split from a hash of the path (not row order/index)."""
    h = int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def class_names(parquet: Path) -> list:
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(parquet).schema_arrow.metadata
    return json.loads(meta[b"huggingface"].decode())["info"]["features"][
        "company_name"
    ]["names"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="dir of LogoDet-3K *.parquet shards")
    ap.add_argument("--out", default="benchmark/eval_real")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--max-area", type=float, default=0.60)
    ap.add_argument("--min-images-per-brand", type=int, default=8)
    ap.add_argument("--detection-only", action="store_true",
                    help="single-class mode: gold = every LogoDet box, no catalog "
                         "mapping. Correct for a single-class 'logo' detector, and "
                         "the only mode that survives LogoDet's annotation density.")
    ap.add_argument("--max-negatives", type=int, default=150)
    ap.add_argument("--max-boxes-per-image", type=int, default=50,
                    help="drop exhaustively-annotated LogoDet images; must match "
                         "scripts/train_logo_detector.py so train and eval agree")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from src.brand_catalog import match_brand

    import cv2
    import pandas as pd

    src = Path(args.parquet)
    shards = sorted(src.glob("*.parquet"))
    if not shards:
        sys.exit(f"FATAL: no *.parquet under {src}")
    out = Path(args.out)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)

    names: list = []
    for shard in shards:
        cn = class_names(shard)
        if not names:
            names = cn

    # ── Pass 1: index path -> [(brand, bbox)] using metadata columns only. ──
    by_image: dict[str, list] = defaultdict(list)
    for shard in shards:
        df = pd.read_parquet(shard, columns=["image_path", "company_name", "bbox"])
        for img, cls, bbox in zip(df["image_path"], df["company_name"], df["bbox"]):
            path = img.get("path") if isinstance(img, dict) else str(img)
            if bbox is None or len(bbox) < 4:
                continue
            brand = match_brand(names[int(cls)] or "") or ""
            by_image[path].append((brand, [float(v) for v in list(bbox)[:4]]))
        print(f"  indexed {shard.name}: {len(df)} boxes / {len(by_image)} images total")

    brand_images: dict[str, set] = defaultdict(set)
    for path, boxes in by_image.items():
        for brand, _ in boxes:
            if brand:
                brand_images[brand].add(path)
    if args.detection_only:
        # Gold is "a logo is here", so every box counts regardless of brand.
        usable = {"__ANY__"}
        print("\nDETECTION-ONLY: gold = all LogoDet boxes (no catalog mapping)")
    else:
        usable = {b for b, imgs in brand_images.items()
                  if len(imgs) >= args.min_images_per_brand}
        print(f"\ncatalog brands with >={args.min_images_per_brand} distinct images: "
              f"{len(usable)}")
        print(f"  {sorted(usable)}")
        if not usable:
            sys.exit("FATAL: no brand has enough images; lower --min-images-per-brand")

    # LogoDet-3K annotates exhaustively (1..1652 boxes/image, median 167). A gold
    # set containing 1500-box shelf photos asks the detector to reproduce
    # LogoDet's annotation density, not to find logos, so cap it here exactly as
    # the trainer does. If these two drift apart the eval measures the wrong task.
    if args.max_boxes_per_image:
        too_dense = {p for p, b in by_image.items()
                     if len(b) > args.max_boxes_per_image}
        for p in too_dense:
            by_image.pop(p, None)
        print(f"\ndropped {len(too_dense)} images with >{args.max_boxes_per_image} boxes")
        brand_images = defaultdict(set)
        for path, boxes in by_image.items():
            for brand, _ in boxes:
                if brand:
                    brand_images[brand].add(path)
        if not args.detection_only:
            usable = {b for b, imgs in brand_images.items()
                      if len(imgs) >= args.min_images_per_brand}
            print(f"catalog brands still evaluable: {len(usable)} -> {sorted(usable)}")
            if not usable:
                sys.exit("FATAL: density cap removed every evaluable brand")

    # Decide negatives up front so pass 2 knows what to write.
    neg_pool = sorted(p for p, b in by_image.items() if not any(x[0] for x in b))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(neg_pool)
    keep_neg = set(neg_pool[: args.max_negatives])
    print(f"negative pool (all logos outside catalog): {len(neg_pool)}, "
          f"keeping {len(keep_neg)}")

    # ── Pass 2: stream shards, decode + write only wanted images. ──
    labels: dict[str, list] = {s: [] for s in SPLITS}
    kept_boxes: Counter = Counter()
    written: set[str] = set()
    for shard in shards:
        df = pd.read_parquet(shard, columns=["image_path"])
        for img in df["image_path"]:
            path = img.get("path") if isinstance(img, dict) else str(img)
            if path in written or path not in by_image:
                continue
            boxes = by_image[path]
            if args.detection_only:
                pos = [("__ANY__", bb) for _, bb in boxes]
            else:
                pos = [(b, bb) for b, bb in boxes if b in usable]
            is_neg = path in keep_neg
            if not pos and not is_neg:
                continue
            raw = img.get("bytes") if isinstance(img, dict) else None
            if not isinstance(raw, (bytes, bytearray)):
                continue
            arr = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if arr is None:
                continue
            h, w = arr.shape[:2]
            kept = []
            for brand, bb in pos:
                x1, y1 = max(0, bb[0]), max(0, bb[1])
                x2, y2 = min(w, bb[2]), min(h, bb[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1) / float(w * h)
                if area < args.min_area or area > args.max_area:
                    continue
                kept.append({"bbox": [x1, y1, x2, y2], "brand": brand})
                kept_boxes[brand] += 1
            if not kept and not is_neg:
                continue
            split = split_for(path, args.val_frac, args.test_frac)
            name = f"{Path(path).stem}.jpg"
            cv2.imwrite(str(out / "images" / split / name), arr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            labels[split].append({
                "image": name,
                "boxes": kept,
                "negative": not kept,
            })
            written.add(path)
        print(f"  wrote from {shard.name}: {len(written)} images total")

    summary = {
        "source": "LogoDet-3K (axonstan/LogoDet-3K, MIT)",
        "shards": [s.name for s in shards],
        "params": {
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "min_area": args.min_area, "max_area": args.max_area,
            "min_images_per_brand": args.min_images_per_brand,
            "max_boxes_per_image": args.max_boxes_per_image,
        },
        "split_sizes": {s: len(labels[s]) for s in SPLITS},
        "positives_per_split": {s: sum(1 for r in labels[s] if r["boxes"]) for s in SPLITS},
        "negatives_per_split": {s: sum(1 for r in labels[s] if r["negative"]) for s in SPLITS},
        "boxes_per_brand": dict(sorted(kept_boxes.items())),
        "mode": "detection_only" if args.detection_only else "brand_attribution",
        "brands_evaluated": len(kept_boxes),
    }
    json.dump(labels, open(out / "labels.json", "w"), indent=1)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/labels.json and images under {out}/images/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
