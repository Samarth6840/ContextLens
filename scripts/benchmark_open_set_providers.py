#!/usr/bin/env python3
"""Compare online logo-identification backends on gold held-out logo crops.

Dry-run is the default and sends no images. Pass --live to upload the labeled
logo crops from the pilot manifest to the selected providers. Never upload full
video frames. Results are written as JSON with accuracy, source-return rate,
latency, request count, errors, and per-example evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.brand_catalog import match_brand  # noqa: E402
from src.openset import _safe_external_error, create_backend  # noqa: E402


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_frame(file_name: str, images_root: Path) -> Path | None:
    for split in ("val", "test"):
        path = images_root / split / file_name
        if path.is_file():
            return path
    return None


def crop_gold_box(frame, bbox_xywh, padding: float = 0.08):
    h, w = frame.shape[:2]
    x, y, bw, bh = (float(v) for v in bbox_xywh)
    px, py = bw * padding, bh * padding
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(w, int(x + bw + px)), min(h, int(y + bh + py))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("gold box is outside the source frame")
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def score_result_titles(results: list, expected_brand: str) -> dict:
    titles = [str(getattr(result, "title", "") or "") for result in results]
    first = match_brand(titles[0]) if titles else None
    matches = [match_brand(title) for title in titles[:5]]
    return {
        "top1_brand": first,
        "top1_hit": first == expected_brand,
        "top5_hit": expected_brand in matches,
        "matched_brands_top5": [brand for brand in matches if brand],
    }


def benchmark_provider(name: str, manifest: list[dict], images_root: Path, live: bool) -> dict:
    try:
        backend = create_backend(name)
    except Exception as exc:
        return {"provider": name, "status": "configuration_error", "error": _safe_external_error(exc)}
    if not backend.available:
        return {"provider": name, "status": "credential_unavailable", "samples": len(manifest)}
    if not live:
        return {
            "provider": name,
            "status": "dry_run",
            "samples": len(manifest),
            "planned_searches": len(manifest),
            "uploads_images": False,
        }

    rows, errors, durations = [], [], []
    request_count = 0
    for sample in manifest:
        frame_path = find_frame(sample["file"], images_root)
        if frame_path is None:
            rows.append({"sample_id": sample["sample_id"], "status": "missing_frame"})
            errors.append("missing_frame")
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            rows.append({"sample_id": sample["sample_id"], "status": "unreadable_frame"})
            errors.append("unreadable_frame")
            continue
        crop = crop_gold_box(frame, sample["bbox_xywh"])
        started = time.perf_counter()
        try:
            # The provider sees only the adjudicated hero crop, never its video
            # frame, expected brand, or neighboring labels.
            results = backend.search_crop(crop)
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            request_count += 1
            result_rows = [result.to_dict() for result in results]
            scored = score_result_titles(results, sample["brand"])
            rows.append({
                "sample_id": sample["sample_id"],
                "brand": sample["brand"],
                "status": "ok",
                "latency_seconds": round(elapsed, 3),
                "source_count": sum(bool(item.get("url")) for item in result_rows),
                **scored,
                "results": result_rows[:10],
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            request_count += 1
            error = _safe_external_error(exc)
            errors.append(error)
            rows.append({
                "sample_id": sample["sample_id"],
                "brand": sample["brand"],
                "status": "error",
                "latency_seconds": round(elapsed, 3),
                "error": error,
            })
            if "API error 429" in error or "API error 503" in error:
                # Stop on provider quota/overload. Sending the remaining
                # samples immediately would waste requests and contaminate
                # the timing/quality comparison with avoidable errors.
                for pending in manifest[len(rows):]:
                    rows.append({
                        "sample_id": pending["sample_id"],
                        "brand": pending["brand"],
                        "status": "not_run_provider_backoff",
                    })
                break

    completed = [row for row in rows if row.get("status") == "ok"]
    source_rows = [row for row in completed if row.get("source_count", 0) > 0]
    top1_hits = sum(bool(row["top1_hit"]) for row in completed)
    top5_hits = sum(bool(row["top5_hit"]) for row in completed)
    return {
        "provider": name,
        "status": "completed" if len(completed) == len(manifest) else "partial",
        "samples": len(manifest),
        "attempted": request_count,
        "requests": request_count,
        "completed": len(completed),
        "top1_accuracy": round(top1_hits / max(1, len(manifest)), 4),
        "top5_accuracy": round(top5_hits / max(1, len(manifest)), 4),
        "conditional_top1_accuracy": round(top1_hits / max(1, len(completed)), 4),
        "conditional_top5_accuracy": round(top5_hits / max(1, len(completed)), 4),
        "source_return_rate": round(len(source_rows) / max(1, len(manifest)), 4),
        "conditional_source_return_rate": round(len(source_rows) / max(1, len(completed)), 4),
        "success_rate": round(len(completed) / max(1, len(manifest)), 4),
        "not_run": sum(row.get("status") == "not_run_provider_backoff" for row in rows),
        "mean_latency_seconds": round(sum(durations) / max(1, len(durations)), 3),
        "errors": errors,
        "per_sample": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmark/logo_product_eval/serpapi_google_lens_pilot.jsonl"))
    parser.add_argument("--images", type=Path, default=Path("benchmark/realvideo_v1/extract/images"))
    parser.add_argument("--providers", default="serpapi,google_vision_web,gemini_grounded")
    parser.add_argument("--max-samples", type=int, default=14)
    parser.add_argument("--live", action="store_true", help="explicitly upload gold logo crops to providers")
    parser.add_argument("--out", type=Path, default=Path("benchmark/results/open_set_provider_pilot.json"))
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    manifest = read_manifest(args.manifest)[:args.max_samples]
    providers = [name.strip() for name in args.providers.split(",") if name.strip()]
    output = {
        "benchmark": "logo_product_eval_pilot",
        "live": bool(args.live),
        "manifest": str(args.manifest),
        "sample_count": len(manifest),
        "image_policy": "gold hero crops only; full video frames are not sent",
        "providers": [benchmark_provider(name, manifest, args.images, args.live) for name in providers],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for provider in output["providers"]:
        print(
            f"{provider['provider']}: {provider['status']} — "
            f"{provider.get('samples', 0)} samples, "
            f"top1={provider.get('top1_accuracy', 'n/a')}, "
            f"top5={provider.get('top5_accuracy', 'n/a')}, "
            f"sources={provider.get('source_return_rate', 'n/a')}"
        )
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
