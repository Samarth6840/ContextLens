"""
Build a per-brand CLIP REFERENCE bank from LogoDet-3K TRAIN parquet shards.

This is the Phase-1 data asset the retrieval benchmark needs: 5-20 clean logo
CROPS per canonical catalog brand, drawn from the real (large) train split so a
brand like SAMSUNG has many distinct reference logos. The held-out TEST queries
come from the separate test split, so this bank is never queried against
itself (honest evaluation).

Writes <out>/reference/<BRAND>/<BRAND>-<n>.png — the exact layout the pipeline
config (`layer1.logo_retrieval.reference_dir`) and scripts/retrieval_benchmark.py
expect.

Honesty guards (repo ethos):
  * Only brands that map to the catalog via src.brand_catalog.match_brand are
    written; anything else is skipped (no fabricating brands).
  * One crop per unique IMAGE per brand by default, so a brand that appears in
    many frames of a single image is not duplicated into N near-identical refs.
  * A global --max-per-brand cap bounds CLIP build time and keeps the bank
    balanced (a 1000-crop brand would dominate/overfit the index).
  * Tiny boxes (< --min-area fraction) are dropped as noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="/tmp/adscene_bench/train")
    p.add_argument("--classes", default="/tmp/adscene_bench/logodet3k_classes.json")
    p.add_argument("--out", default="benchmark/reference_logos_bank/reference")
    p.add_argument("--shards", type=int, default=0, help="0 = all shards in dir")
    p.add_argument("--max-per-brand", type=int, default=40)
    p.add_argument("--one-per-image", action="store_true", default=True,
                   help="take at most one crop per unique image per brand")
    p.add_argument("--min-area", type=float, default=0.004,
                   help="min bbox area as fraction of image")
    args = p.parse_args()

    from src.brand_catalog import match_brand

    import cv2
    import pandas as pd

    shards = sorted(Path(args.train_dir).glob("*.parquet"))
    if args.shards:
        shards = shards[: args.shards]
    if not shards:
        sys.exit(f"FATAL: no parquet shards in {args.train_dir}")
    print(f"Reading {len(shards)} train shard(s) from {args.train_dir}")

    classes = json.load(open(args.classes))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # brand -> list of (image_path, crop). one-per-image enforced via a set of
    # (brand, image_path) so duplicate instances of the same logo are skipped.
    collected = defaultdict(list)
    seen_img = set()
    for shard in shards:
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            ip = row["image_path"]
            path = ip.get("path") if isinstance(ip, dict) else str(ip)
            idxs = {int(row["company_name"])}
            brand = ""
            for i in idxs:
                b = match_brand(classes.get(str(i), ""))
                if b:
                    brand = b.upper()
                    break
            if not brand:
                continue
            if len(collected[brand]) >= args.max_per_brand:
                continue
            key = (brand, path)
            if args.one_per_image and key in seen_img:
                continue
            raw = ip.get("bytes") if isinstance(ip, dict) else None
            img = None
            if isinstance(raw, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            bbox = list(row["bbox"])[:4] if row["bbox"] is not None else None
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) * (y2 - y1) / (h * w) < args.min_area:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            seen_img.add(key)
            collected[brand].append((path, crop))
        print(f"  {shard.name}: done ({sum(len(v) for v in collected.values())} crops so far)")

    n_written = 0
    for brand in sorted(collected):
        d = out / brand
        d.mkdir(parents=True, exist_ok=True)
        for n, (path, crop) in enumerate(collected[brand]):
            cv2.imwrite(str(d / f"{brand}-{n:03d}.png"), crop)
            n_written += 1
        print(f"  {brand}: {len(collected[brand])} reference crops")
    print(f"\nWrote {n_written} reference crops to {out}")


if __name__ == "__main__":
    main()
