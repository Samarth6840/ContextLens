"""
Held-out CLIP logo-retrieval benchmark (Phase 1+2 of the brand-resolution
remediation). Measures whether CLIP image retrieval can turn a detected logo
REGION into a correct BRAND when OCR finds no wordmark (the icon-only / stylized
case where YOLO-World confidently mislabels, e.g. "SUPREME logo" on a Samsung).

Honest evaluation design (no closed-set inflation):
  * REFERENCE bank  = real per-brand logo crops, train LogoDet split
  * TEST queries    = real per-brand logo crops, test  LogoDet split
  * A test crop's logo is NEVER in the reference bank (different split), so a
    top-1 hit is a genuine generalization result, not a self-match.

This script scores CLIP retrieval alone (top-1 brand per held-out crop). The
full production fusion (OCR > CLIP retrieval > class-label) that consumes this
index is exercised end-to-end by scripts/resolution_benchmark.py instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("retrieval_bench")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_bank(ref_dir: Path):
    """reference bank: <ref_dir>/<BRAND>/*.png -> list of (brand, img)."""
    bank = defaultdict(list)
    if not ref_dir.is_dir():
        sys.exit(f"FATAL: reference bank dir not found: {ref_dir}")
    for img_path in sorted(ref_dir.glob("*/*.png")):
        import cv2

        brand = img_path.parent.name.upper()
        img = cv2.imread(str(img_path))
        if img is not None:
            bank[brand].append((img_path, img))
    return dict(bank)


def extract_test_queries(parquet, classes_json, subset):
    """Load held-out test crops as (brand, crop_img) from a LogoDet parquet,
    keeping only brands in `subset`. Splits nothing here — the split is done at
    extraction time by the caller (train ref vs test query)."""
    import cv2
    import pandas as pd

    classes = json.load(open(classes_json))
    from src.brand_catalog import match_brand

    df = pd.read_parquet(parquet)
    queries = []
    seen = set()
    for _, row in df.iterrows():
        ip = row["image_path"]
        path = ip.get("path") if isinstance(ip, dict) else str(ip)
        brand = (match_brand(classes.get(str(int(row["company_name"])), "")) or "").upper()
        if not brand or brand not in subset:
            continue
        raw = ip.get("bytes") if isinstance(ip, dict) else None
        if isinstance(raw, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            img = None
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = list(row["bbox"])[:4] if row["bbox"] is not None else None
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        if (x2 - x1) * (y2 - y1) / (h * w) < 0.002:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        key = (path, row.name)
        if key in seen:
            continue
        seen.add(key)
        queries.append({"path": f"{path}#{row.name}", "brand": brand, "crop": crop})
    return queries


def run_retrieval(ref_dir: Path, parquet: Path, classes_json, device: str, limit: int):
    from src.layer1.logo_retrieval import LogoRetrievalIndex

    bank = load_bank(ref_dir)
    subset = set(bank)
    logger.info("Reference bank brands: %s", {b: len(v) for b, v in bank.items()})

    index = LogoRetrievalIndex()
    if device:
        index._device = device
    for brand, imgs in bank.items():
        for _, img in imgs:
            index.add_brand(brand, [img])
    logger.info("CLIP index built (%d crops, %d brands)", index._embeddings.shape[0], len(index.brands))

    queries = extract_test_queries(parquet, classes_json, subset)
    if limit:
        queries = queries[:limit]
    logger.info("Held-out test queries: %d", len(queries))

    per_brand = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    confs = []
    errors = []
    for q in queries:
        res = index.query(q["crop"], top_k=3)
        top1 = res[0][0] if res else None
        correct = top1 == q["brand"]
        per_brand[q["brand"]]["total"] += 1
        if correct:
            per_brand[q["brand"]]["top1"] += 1
            confs.append(res[0][1])
        if any(b == q["brand"] for b, s in res[:3]):
            per_brand[q["brand"]]["top3"] += 1
        if not correct:
            errors.append({
                "query": q["path"], "gt_brand": q["brand"],
                "predicted": top1,
                "top3": [(b, s) for b, s in res[:3]],
            })

    total = sum(v["total"] for v in per_brand.values())
    top1 = sum(v["top1"] for v in per_brand.values())
    top3 = sum(v["top3"] for v in per_brand.values())

    agg = {
        "test_crops": total,
        "top1_accuracy": round(top1 / total, 3) if total else 0.0,
        "top3_accuracy": round(top3 / total, 3) if total else 0.0,
        "mean_top1_similarity": round(sum(confs) / len(confs), 3) if confs else None,
        "per_brand": {
            b: {
                "total": v["total"],
                "top1_accuracy": round(v["top1"] / v["total"], 3) if v["total"] else None,
                "top3_accuracy": round(v["top3"] / v["total"], 3) if v["total"] else None,
            }
            for b, v in sorted(per_brand.items())
        },
    }
    return agg, errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", default="benchmark/reference_logos_bank/reference")
    p.add_argument("--test-parquet", default="/tmp/adscene_bench/test-00000-of-00002.parquet")
    p.add_argument("--classes", default="/tmp/adscene_bench/logodet3k_classes.json")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="benchmark/results")
    p.add_argument("--diag", action="store_true", help="print per-query misclassifications")
    args = p.parse_args()

    agg, errors = run_retrieval(
        Path(args.ref_dir), Path(args.test_parquet), args.classes, args.device, args.limit
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"retrieval_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_data = dict(agg)
    out_data["errors"] = errors
    (out_dir / out_file.name).write_text(json.dumps(out_data, indent=2))
    print(json.dumps(agg, indent=2))
    if args.diag:
        print("\nMisclassified queries:")
        for e in errors:
            print(f"  {e['gt_brand']:12s} -> {str(e['predicted']):12s} "
                  f"top3={[(b, s) for b, s in e['top3']]}  {e['query']}")
    print(f"\nWrote {(out_dir / out_file.name)}")


if __name__ == "__main__":
    main()
