#!/usr/bin/env python3
"""Build a deterministic provider-evaluation pilot from human-labeled holdout.

Only gold hero boxes from the frozen val/test videos are included. Selection
keeps one highest-resolution example per brand so a small paid-provider trial
has broad brand diversity without uploading full frames or repeated near-dup
frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(labels_path: Path, splits_path: Path) -> list[dict]:
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    video_split = {
        str(video): info["split"]
        for video, info in (splits.get("videos") or {}).items()
    }
    best: dict[str, dict] = {}
    for line_no, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        video = str(row.get("video") or row.get("file", "").split("_", 1)[0])
        split = video_split.get(video)
        if split not in {"val", "test"} or row.get("verdict") != "logo":
            continue
        for box in row.get("boxes") or []:
            brand = str(box.get("brand") or "").strip().upper()
            if not brand or not box.get("hero") or not box.get("on", True):
                continue
            try:
                width, height = float(box["w"]), float(box["h"])
                x, y = float(box["x"]), float(box["y"])
            except (KeyError, TypeError, ValueError):
                continue
            area = max(0.0, width) * max(0.0, height)
            if area <= 0:
                continue
            candidate = {
                "sample_id": f"{video}:{Path(row['file']).stem}:{brand}",
                "file": row["file"],
                "video": video,
                "split": split,
                "brand": brand,
                "bbox_xywh": [x, y, width, height],
                "bbox_area": round(area, 2),
                "label_provenance": box.get("src", "human"),
                "label_line": line_no,
            }
            previous = best.get(brand)
            if previous is None or candidate["bbox_area"] > previous["bbox_area"]:
                best[brand] = candidate
    return [best[brand] for brand in sorted(best)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("benchmark/realvideo_v1/labels.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("benchmark/realvideo_v1/splits.json"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("benchmark/logo_product_eval/serpapi_google_lens_pilot.jsonl"),
    )
    args = parser.parse_args()
    rows = build_manifest(args.labels, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} held-out brand crops to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
