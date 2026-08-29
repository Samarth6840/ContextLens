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


def run(refdir: str, device: str, limit: int):
    from src.brand_catalog import build_text_queries
    from src.layer1.logo_detector import create_logo_detector
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
    resolver = BrandResolver(ocr_extractor=ocr, class_confidence=0.40, crop_scale=2.0)

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
    n = len(rows)
    detected = [r for r in rows if r["n_regions"] > 0]
    resolved_any = [r for r in rows if r["resolved"]]
    correct = []

    per_brand: dict = {}
    for r in rows:
        key = r["gt_brand"]
        b = per_brand.setdefault(
            key, {"images": 0, "detected": 0, "resolved": 0, "correct": 0, "confs": []}
        )
        b["images"] += 1
        b["detected"] += 1 if r["n_regions"] > 0 else 0

        best = None
        for d in r["resolved"]:
            cand = (d.get("brand"), float(d.get("confidence", 0.0)))
            if cand[0]:
                if best is None or cand[1] > best[1]:
                    best = cand
        if best:
            b["resolved"] += 1
            if (best[0] or "").upper() == r["gt_brand"]:
                b["correct"] += 1
                b["confs"].append(best[1])
                correct.append(r)

    total_correct = sum(b["correct"] for b in per_brand.values())
    total_resolved = sum(b["resolved"] for b in per_brand.values())

    aggregate = {
        "images": n,
        "detected_rate": round(len(detected) / n, 3) if n else 0.0,
        "resolve_rate": round(len(resolved_any) / n, 3) if n else 0.0,
        "resolution_accuracy": round(total_correct / len(resolved_any), 3)
        if resolved_any else 0.0,
        "brand_accuracy": round(total_correct / total_resolved, 3)
        if total_resolved else 0.0,
        "mean_correct_conf": round(
            sum(c for b in per_brand.values() for c in b["confs"]) / total_correct, 3
        ) if total_correct else 0.0,
        "latency_total_s": round(elapsed, 3),
        "latency_ms_per_image": round(elapsed / n * 1000, 1) if n else None,
        "per_brand": {
            k: {
                "images": v["images"],
                "detected": v["detected"],
                "resolved": v["resolved"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["resolved"], 3)
                if v["resolved"] else 0.0,
                "mean_correct_conf": round(sum(v["confs"]) / len(v["confs"]), 3)
                if v["confs"] else 0.0,
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
                        "ocr_text": d.get("ocr_text")}
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
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="benchmark/results")
    args = p.parse_args()

    if not Path(args.refdir).is_dir():
        sys.exit(f"FATAL: reference dir not found: {args.refdir}")

    result = run(args.refdir, args.device, args.limit)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"resolution_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
    print("\nPer-image:")
    for it in result["per_image"]:
        mark = "✓" if it["correct"] else ("—" if it["resolved"] else "✗")
        print(
            f"  {it['file']:18s} gt={it['gt']:10s} regions={it['n_regions']} "
            f"resolved={str(it['resolved'].get('brand') if it['resolved'] else None):12s} "
            f"conf={it['resolved'].get('confidence') if it['resolved'] else None}  {mark}"
        )
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
