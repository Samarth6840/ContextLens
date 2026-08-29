"""
End-to-end BRAND RESOLUTION benchmark on the real reference-logo images.

Measures the value delivered by the brand-resolution fixes: from a raw video
frame (here, a real product-logo image) -> logo region detection -> brand name.

Unlike scripts/logo_detection_benchmark.py (which measures the raw LOGO-REGION
backend in isolation, before brand resolution), this benchmark drives the ACTUAL
production stage that was fixed:

    YOLO-World logo detector  ->  BrandResolver (confidence gate + crop-OCR)

and reports, per brand and in aggregate:

  * resolve_rate      — fraction of images where ANY logo region was found
  * resolution_accuracy — of images that produced a resolved brand, the fraction
                          resolved to the CORRECT ground-truth brand
  * brand_accuracy    — per-brand correct/attempted
  * mean_conf         — mean confidence of CORRECT resolutions (signal strength)
  * sample breakdown  — per-image: GT brand, resolved brand, confidence, OCR hit

This is intentionally a SMALL, curated, fail-closed check (the repo's ethos): it
does not fabricate a pseudo-dataset, and reports honestly when a model/checkpoint
is missing. It operates on real files under benchmark/reference_logos/.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("resolution_bench")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def ground_truth_from_name(fname: str) -> str:
    """Brand is the token before the first '_' in the file name, canonicalized."""
    brand = Path(fname).name.split("_")[0].upper().strip()
    return brand


def load_images(refdir: str):
    images = []
    for f in sorted(glob.glob(str(Path(refdir) / "*.png"))):
        import cv2

        img = cv2.imread(f)
        if img is None:
            logger.warning("Skipping unreadable image %s", f)
            continue
        images.append({"path": f, "image": img, "gt_brand": ground_truth_from_name(f)})
    return images


def run(refdir: str, device: str, limit: int, retrieval_dir: str = None):
    from src.brand_catalog import build_text_queries
    from src.layer1.logo_detector import create_logo_detector
    from src.layer1.logo_retrieval import LogoRetrievalIndex
    from src.layer1.ocr import OCRExtractor
    from src.layer2.brand_resolver import BrandResolver

    # ── Real production path ──────────────────────────────────────────────
    logger.info("Loading logo detector (YOLO-World)...")
    queries = ["brand logo", "company logo", "text logo", "product logo", "logo"]
    for q in build_text_queries():
        if q not in queries:
            queries.append(q)
    detector = create_logo_detector(
        backend="yolo_world",
        model_name="yolov8s-worldv2.pt",
        confidence_threshold=0.30,
        device=device,
        text_queries=queries,
    )
    logger.info("Loading PaddleOCR...")
    ocr = OCRExtractor(lang="en", use_angle_cls=True, det_db_thresh=0.3, rec_batch_num=6)

    retrieval_index = None
    if retrieval_dir and Path(retrieval_dir).is_dir():
        logger.info("Loading CLIP logo-retrieval index from %s", retrieval_dir)
        retrieval_index = LogoRetrievalIndex.build_from_dir(retrieval_dir, device="cpu")
        logger.info("  index: %d crops / %d brands",
                    retrieval_index._embeddings.shape[0], len(retrieval_index.brands))
    resolver = BrandResolver(
        ocr_extractor=ocr,
        class_confidence=0.40,
        crop_scale=2.0,
        retrieval_index=retrieval_index,
        retrieval_min_similarity=0.22,
    )

    images = load_images(refdir)
    if limit:
        images = images[:limit]

    rows = []
    t0 = time.monotonic()
    for img in images:
        dets = detector.detect_batch([img["image"]], text_queries=queries)[0]
        resolved = resolver.resolve([dets], [img["image"]])[0] if dets else []
        row = {
            "file": Path(img["path"]).name,
            "gt_brand": img["gt_brand"],
            "n_regions": len(dets),
            "resolved": resolved,
        }
        rows.append(row)
    elapsed = time.monotonic() - t0

    # ── Aggregate ────────────────────────────────────────────────────────
    # Metric conventions (made explicit so the denominators can't silently
    # disagree):
    #   * a "region" is a YOLO logo-box proposal handed to the resolver.
    #   * a "brand resolve attempt" is a region whose resolver output carries a
    #     NON-None `brand` (i.e. we committed to a brand identity for it).
    #   * resolution_accuracy and brand_accuracy share that same denominator,
    #     differing only in granularity:
    #       - brand_accuracy: per-REGION  (correct resolves / brand attempts)
    #       - resolution_accuracy: per-IMAGE (best region's resolve vs GT),
    #         correct / images-that-resolved-a-brand. Reported with numerator &
    #         denominator visible so it can be audited.
    n_images = len(rows)
    images_detected = [r for r in rows if r["n_regions"] > 0]

    total_regions = sum(r["n_regions"] for r in rows)
    brand_attempts = 0
    correct_regions = 0
    images_with_brand = 0
    images_correct = 0
    correct_confidences: List[float] = []
    resolution_sources: Counter = Counter()

    # Per-region stats keyed by (file, GT) for the sample breakdown.
    per_brand: dict = {}
    for r in rows:
        key = r["gt_brand"]
        b = per_brand.setdefault(
            key, {"images": 0, "regions": 0, "attempts": 0,
                  "correct": 0, "confs": []}
        )
        b["images"] += 1
        b["regions"] += r["n_regions"]

        resolved = r["resolved"]
        # Best (highest-confidence) brand-bearing region for THIS image.
        best = None
        for d in resolved:
            if not d.get("brand"):
                continue
            brand_attempts += 1
            resolution_sources[d.get("resolution_source", "class_label")] += 1
            dconf = float(d.get("confidence", 0.0))
            if best is None or dconf > best[1]:
                best = (d["brand"], dconf)

        if best:
            images_with_brand += 1
            is_correct = best[0].upper() == r["gt_brand"]
            if is_correct:
                images_correct += 1
            # Per-REGION correctness: count every region resolved to GT.
            for d in resolved:
                if d.get("brand") and d["brand"].upper() == r["gt_brand"]:
                    b["correct"] += 1
                    correct_regions += 1
                    correct_confidences.append(float(d.get("confidence", 0.0)))
                    b["confs"].append(float(d.get("confidence", 0.0)))
            b["attempts"] += len([d for d in resolved if d.get("brand")])

    aggregate = {
        "images": n_images,
        "regions_total": total_regions,
        "regions_detected_rate": round(len(images_detected) / n_images, 3)
        if n_images else 0.0,
        # Per-region brand metrics (shared denominator = brand resolve attempts)
        "brand_attempts": brand_attempts,
        "brand_correct": correct_regions,
        "brand_accuracy": round(correct_regions / brand_attempts, 3)
        if brand_attempts else 0.0,
        "resolution_sources": dict(sorted(resolution_sources.items())),
        # Per-image resolution metrics (denominator = images that resolved a brand)
        "images_resolved_brand": images_with_brand,
        "images_correct": images_correct,
        "resolution_accuracy": round(images_correct / images_with_brand, 3)
        if images_with_brand else 0.0,
        # Confidence of CORRECT regions — note this is the detector/class
        # confidence, NOT a dedicated match-conf; see Phase 2 (CLIP similarity).
        "mean_correct_detection_conf": round(
            sum(correct_confidences) / len(correct_confidences), 3
        ) if correct_confidences else 0.0,
        "latency_total_s": round(elapsed, 3),
        "latency_ms_per_image": round(elapsed / n_images * 1000, 1)
        if n_images else None,
        "per_brand": {
            k: {
                "images": v["images"],
                "regions": v["regions"],
                "attempts": v["attempts"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["attempts"], 3)
                if v["attempts"] else 0.0,
                "mean_correct_detection_conf": round(
                    sum(v["confs"]) / len(v["confs"]), 3
                ) if v["confs"] else 0.0,
            }
            for k, v in sorted(per_brand.items())
        },
    }

    sample = []
    for r in rows:
        best = None
        for d in r["resolved"]:
            if d.get("brand") and (best is None or d["confidence"] > best["confidence"]):
                best = {"brand": d["brand"], "confidence": round(d["confidence"], 3),
                        "ocr_text": d.get("ocr_text"),
                        "resolution_source": d.get("resolution_source"),
                        "retrieval_top3": d.get("retrieval_top3")}
        sample.append({
            "file": r["file"],
            "gt": r["gt_brand"],
            "n_regions": r["n_regions"],
            "resolved": best,
            "correct": bool(best and best["brand"].upper() == r["gt_brand"]),
        })

    return {
        "script": "resolution_benchmark.py",
        "refdir": refdir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "aggregate": aggregate,
        "per_image": sample,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refdir", default="benchmark/reference_logos")
    p.add_argument("--retrieval-dir", default="benchmark/reference_logos_bank/reference")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="benchmark/results")
    args = p.parse_args()

    if not Path(args.refdir).is_dir():
        sys.exit(f"FATAL: reference dir not found: {args.refdir}")

    result = run(args.refdir, args.device, args.limit,
                 retrieval_dir=None if args.no_retrieval else args.retrieval_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"resolution_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
    print("\nPer-image:")
    for it in result["per_image"]:
        mark = "✓" if it["correct"] else ("—" if it["resolved"] else "✗")
        r = it["resolved"]
        src = r.get("resolution_source") if r else None
        print(
            f"  {it['file']:18s} gt={it['gt']:10s} regions={it['n_regions']} "
            f"resolved={str(r.get('brand') if r else None):12s} src={str(src):14s} "
            f"conf={r.get('confidence') if r else None}  {mark}"
        )
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
