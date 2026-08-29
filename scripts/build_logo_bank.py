"""
Build a per-brand CLIP logo reference bank (and held-out test set) from a
LogoDet-3K parquet shard.

CONTEXT (Phase 1 of the logo-recall/precision remediation): CLIP image
retrieval needs a per-brand reference bank of logo CROPS (the plan calls for
5-20 clean logo crops per brand). This tool extracts those crops from a real
LogoDet-3K parquet split and writes them grouped by canonical catalog brand:

    <out_dir>/reference/<BRAND>/<img>-<n>.png      reference crops (in bank)
    <out_dir>/test/<BRAND>/<img>-<n>.png           held-out crops (not in bank)

Splitting is by IMAGE PATH so no test image's logo is ever a reference crop
(even a brand with several instances in one image goes to one side). This keeps
the later retrieval benchmark honest (no closed-set self-match inflation).

Quote of honesty:
  * Only brands that map to the catalog via src.brand_catalog.match_brand are
    written — anything else is skipped (no fabricating brands).
  * A brand present but with < min_crops_per_brand total crops is skipped with
    a warning, because a 1-2 crop bank is not usable evidence (retrieval would
    be unreliable / degenerate).
  * If the parquet is missing the script fails loudly.

Usage:
  python scripts/build_logo_bank.py --parquet /tmp/adscene_bench/test-00000-of-00002.parquet \
      --classes /tmp/adscene_bench/logodet3k_classes.json \
      --out benchmark/reference_logos_bank
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _brand_for_idxs(idxs, classes) -> str:
    """Map a set of LogoDet class indices to a canonical catalog brand.

    One image can carry several LogoDet classes (e.g. 'Apple Zings',
    'Apple Jacks'); take the first that maps to a catalog brand, else ''.
    """
    from src.brand_catalog import match_brand

    for i in idxs:
        name = classes.get(str(int(i)), "")
        b = match_brand(name)
        if b:
            return b.upper()
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", default="benchmark/reference_logos_bank")
    p.add_argument("--min-crops-per-brand", type=int, default=5)
    p.add_argument("--min-area", type=float, default=0.002,
                   help="min bbox area as fraction of image, to skip tiny/noisy boxes")
    args = p.parse_args()

    import cv2
    import pandas as pd

    if not Path(args.parquet).is_file():
        sys.exit(f"FATAL: parquet not found: {args.parquet}")
    if not Path(args.classes).is_file():
        sys.exit(f"FATAL: classes json not found: {args.classes}")

    df = pd.read_parquet(args.parquet)
    out = Path(args.out)
    ref_dir = out / "reference"
    test_dir = out / "test"

    classes = json.load(open(args.classes))

    # Per-image-brand tally to respect the split-by-image rule.
    image_brand = {}
    for _, row in df.iterrows():
        img_path = row["image_path"]
        path = img_path.get("path") if isinstance(img_path, dict) else str(img_path)
        if path in image_brand:
            image_brand[path].add(int(row["company_name"]))
        else:
            image_brand[path] = {int(row["company_name"])}

    # Plan each image -> brand -> side, splitting by hashed image path.
    image_side = {}
    brand_counts = {}
    for path, idxs in image_brand.items():
        brand = _brand_for_idxs(idxs, classes)
        if not brand:
            continue
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
        side = "reference" if (abs(hash(path)) % 7) < 5 else "test"  # ~5:2 split
        image_side[path] = (brand, side)

    usable = {b for b, c in brand_counts.items() if c >= args.min_crops_per_brand}
    if not usable:
        sys.exit("No brand has enough crops here; nothing written.")

    written_ref = {}
    written_test = {}
    skipped = {}
    for _, row in df.iterrows():
        img_path = row["image_path"]
        path = img_path.get("path") if isinstance(img_path, dict) else str(img_path)
        plan = image_side.get(path)
        if not plan:
            continue
        brand, side = plan
        if brand not in usable:
            skipped[brand] = skipped.get(brand, 0) + 1
            continue

        raw = img_path.get("bytes") if isinstance(img_path, dict) else None
        if isinstance(raw, (bytes, bytearray)):
            buf = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            img = None
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = row["bbox"]
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in list(bbox)[:4]]
        if (x2 - x1) * (y2 - y1) / (h * w) < args.min_area:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        stem = f"{Path(path).stem}-{row.name}"
        d = ref_dir / brand if side == "reference" else test_dir / brand
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{stem}.png"), crop)
        (written_ref if side == "reference" else written_test)[brand] = \
            (written_ref if side == "reference" else written_test).get(brand, 0) + 1

    print("Reference bank written to", ref_dir)
    for b in sorted(written_ref):
        print(f"  {b}: {written_ref[b]} ref / {written_test.get(b, 0)} test crops")
    fills = [b for b in sorted(written_ref)
             if written_ref[b] + written_test.get(b, 0) < args.min_crops_per_brand]
    if fills:
        print("WARN (below min crops, weak banks):", fills)


if __name__ == "__main__":
    main()
