"""
Logo-detection benchmark runner (Part B remediation).

Loads a REAL held-out logo-detection dataset (LogoDet-3K test split, parquet)
and evaluates every implemented detection paradigm on the same images with the
same object-detection metrics (IoU@0.5 matching, Precision/Recall/F1, mAP@0.5).

Backends (see scripts/logo_bench_backends.py):
  * yolo_world            — production zero-shot detector (YOLO-World)
  * region_proposal_clip  — selective-search + CLIP zero-shot classification
  * sift_match            — SIFT keypoint matching against a reference bank

Usage:
  python scripts/logo_detection_benchmark.py \
      --parquet benchmark/test-00000-of-00002.parquet \
      --limit 40 \
      --backends yolo_world region_proposal_clip sift_match \
      --output benchmark/results

Notes on honesty:
  * Only REAL parquet data is evaluated. If the dataset is missing the script
    fails loudly — no synthetic stand-ins.
  * Brand names in LogoDet-3K are the 3000-class English names (e.g. "Adidas");
    we map them to catalog canonical names for matching (via match_brand).
  * SIFT needs a reference bank; if none exists it degrades to zero detections
    and reports that honestly.
  * mAP@0.5 integrates the confidence-sweep PR curve (see logo_bench_metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("logo_benchmark")

from scripts.logo_bench_metrics import evaluate  # noqa: E402


def load_logodet_parquet(path: str, limit: int, classes_json: str):
    """Load LogoDet-3K test parquet rows into a list of image dicts.

    The parquet carries `company_name` as a class index and `bbox` as a numpy
    [x1,y1,x2,y2] array; the index->name map is provided in the dataset README
    (kept at classes_json). Labels are resolved to catalog canonical brands.
    """
    import json

    import pandas as pd

    from src.brand_catalog import match_brand

    classes = json.load(open(classes_json))
    df = pd.read_parquet(path)

    def _path(v):
        return v.get("path") if isinstance(v, dict) else v

    df["_path"] = df["image_path"].apply(_path)
    logger.info("Parquet rows: %d, unique images: %d", len(df), df["_path"].nunique())

    images = []
    for path, group in df.groupby("_path"):
        raw = group.iloc[0]["image_path"]
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        img = None
        if isinstance(raw, (bytes, bytearray)):
            import cv2

            buf = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt = []
        for _, row in group.iterrows():
            class_idx = int(row["company_name"])
            name = classes.get(str(class_idx)) or classes.get(class_idx)
            brand = match_brand(name or "")
            if not brand:
                continue
            bbox = row["bbox"]
            if bbox is None or len(bbox) < 4:
                continue
            box = [float(v) for v in np.asarray(bbox).flatten()[:4]]
            gt.append({"bbox": box, "brand": brand})
        if not gt:
            continue
        images.append({"image": img, "gt": gt, "id": path})
        if len(images) >= limit:
            break
    logger.info(
        "Loaded %d images (%d GT boxes, unique brands: %d)",
        len(images),
        sum(len(i["gt"]) for i in images),
        len({g["brand"] for i in images for g in i["gt"]}),
    )
    return images


def text_queries_for_catalog() -> list:
    """Per-brand text prompts for zero-shot backends."""
    from src.brand_catalog import build_text_queries

    q = ["brand logo", "company logo", "text logo", "product logo", "logo"]
    for bq in build_text_queries():
        if bq not in q:
            q.append(bq)
    return q


def make_backend(name: str, device: str):
    from scripts.logo_bench_backends import (
        RegionProposalCLIPBackend,
        SIFTLogoMatcherBackend,
        YOLOWorldBackend,
    )

    if name == "yolo_world":
        return YOLOWorldBackend(device=device)
    if name == "region_proposal_clip":
        return RegionProposalCLIPBackend(device=device)
    if name == "sift_match":
        return SIFTLogoMatcherBackend(reference_bank_dir="benchmark/reference_logos")
    raise ValueError(f"unknown backend: {name}")


def run(parquet: str, classes_json: str, limit: int, backends: list, device: str, output: str):
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_logodet_parquet(parquet, limit, classes_json)
    queries = text_queries_for_catalog()

    results = {
        "dataset": {
            "name": "LogoDet-3K test split (axonstan mirror)",
            "parquet": parquet,
            "n_images": len(images),
            "n_gt_boxes": sum(len(i["gt"]) for i in images),
            "gt_brand_counts": _count_brands(images),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backends": {},
    }

    for backend_name in backends:
        logger.info("=== Backend: %s ===", backend_name)
        backend = make_backend(backend_name, device)
        preds = []
        t0 = time.monotonic()
        total_boxes = 0
        for idx, img in enumerate(images):
            dets = backend.detect(img["image"], queries)
            total_boxes += len(dets)
            # Keep brand-agnostic detections but remember which brands matched.
            preds.append(dets)
            if (idx + 1) % 10 == 0:
                logger.info("  %d/%d images, %d boxes so far", idx + 1, len(images), total_boxes)
        elapsed = time.monotonic() - t0

        eval_input = [
            {"gt": img["gt"], "pred": preds[i]} for i, img in enumerate(images)
        ]
        metrics = evaluate(eval_input)
        metrics["latency_s_total"] = round(elapsed, 3)
        metrics["latency_ms_per_image"] = round(elapsed / len(images) * 1000, 1) if images else None
        metrics["boxes_per_image"] = round(total_boxes / len(images), 2) if images else None
        results["backends"][backend_name] = metrics
        logger.info("  %s: %s", backend_name, json.dumps(metrics["aggregate"], default=str))

    out_file = out_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote benchmark results to %s", out_file)
    return results


def _count_brands(images) -> dict:
    counts = {}
    for img in images:
        for g in img["gt"]:
            counts[g["brand"]] = counts.get(g["brand"], 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def main():
    parser = argparse.ArgumentParser(description="Logo detection benchmark")
    parser.add_argument("--parquet", required=True, help="Path to LogoDet-3K test parquet")
    parser.add_argument(
        "--classes",
        default="/tmp/adscene_bench/logodet3k_classes.json",
        help="JSON mapping class index -> brand name (from dataset README)",
    )
    parser.add_argument("--limit", type=int, default=40, help="Max images to evaluate")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["yolo_world"],
        help="Backends to run",
    )
    parser.add_argument("--device", default=None, help="torch device override")
    parser.add_argument("--output", default="benchmark/results")
    args = parser.parse_args()

    if not Path(args.parquet).is_file():
        sys.exit(f"FATAL: parquet not found: {args.parquet} — download a real LogoDet-3K test shard first.")
    run(args.parquet, args.classes, args.limit, args.backends, args.device, args.output)


if __name__ == "__main__":
    main()
