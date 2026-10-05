"""Build a single-class YOLO logo corpus from OpenLogo (QMUL mirror on HF).

Why this dataset: LogoDet-3K (the only local mirror) is logo-saturated, median
186 boxes per image, so training on it teaches "everything is a logo". OpenLogo
is 27,083 web photographs with a median of 1 box per image and a median box
area of 0.0074 (82px at 960) - sparse frames with small marks, which is the
distribution ContextLens actually sees.

Two things this script is careful about:

1. Contamination. OpenLogo and benchmark/eval_real are both web-scraped, and
   15 of the 73 held-out test images also occur in OpenLogo (verified by 64-bit
   dHash, far above chance). Any colliding image is dropped, so the held-out
   benchmark stays clean.

2. Space. Each parquet shard is deleted as soon as it is converted, so peak
   usage is one shard plus the growing image set rather than the whole 4.2GB
   download.

Box format in OpenLogo is XYWH and arrives as an object array of per-box
arrays, unlike LogoDet-3K's flat XYXY rows.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import importlib.util as ilu

_spec = ilu.spec_from_file_location(
    "tld", Path(__file__).resolve().parent / "train_logo_detector.py"
)
tld = ilu.module_from_spec(_spec)
_spec.loader.exec_module(tld)


def dhash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return "".join("1" if b else "0"
                   for b in (small[:, 1:] > small[:, :-1]).flatten())


def eval_hashes(eval_set: Path) -> set:
    """Every dHash in the benchmark, so no split can be leaked into training."""
    out = set()
    for split in ("train", "val", "test"):
        d = eval_set / "images" / split
        if not d.exists():
            continue
        for p in d.glob("*.jpg"):
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                out.add(dhash(g))
    return out


def boxes_of(ob) -> np.ndarray:
    if ob is None:
        return np.zeros((0, 4))
    return np.stack([np.asarray(b, dtype=float).ravel()[:4] for b in ob["bbox"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="/tmp/logodl/openlogo/data")
    ap.add_argument("--out", default="weights/logo_corpus")
    ap.add_argument("--eval-set", default="benchmark/eval_real")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-area", type=float, default=0.0005)
    ap.add_argument("--max-area", type=float, default=1.0)
    ap.add_argument("--max-boxes-per-image", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--delete-parquet", action="store_true",
                    help="delete each shard after converting (space)")
    args = ap.parse_args()

    src = Path(args.parquet_dir)
    shards = sorted(src.glob("*.parquet"))
    if not shards:
        sys.exit(f"FATAL: no *.parquet under {src}")
    block = eval_hashes(Path(args.eval_set))
    print(f"blocklist: {len(block)} benchmark dHashes")

    out = Path(args.out)
    counts = {s: 0 for s in ("train", "val", "test")}
    boxes_written = 0
    dropped_contam = dropped_cap = dropped_area = 0
    seen = 0

    for si, shard in enumerate(shards, 1):
        df = pd.read_parquet(shard, columns=["image", "width", "height", "objects"])
        for img, ob, w, h in zip(df["image"], df["objects"], df["width"], df["height"]):
            if args.limit and seen >= args.limit:
                break
            seen += 1
            raw = img.get("bytes") if isinstance(img, dict) else None
            if not isinstance(raw, (bytes, bytearray)):
                continue
            arr = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            hh, ww = arr.shape[:2]
            if dhash(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)) in block:
                dropped_contam += 1
                continue
            try:
                b = boxes_of(ob)
            except Exception:
                continue
            if args.max_boxes_per_image and len(b) > args.max_boxes_per_image:
                dropped_cap += 1
                continue
            A = float(ww * hh)
            lines, bad = [], False
            for x, y, bw, bh in b:                    # XYWH
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(ww), x + bw), min(float(hh), y + bh)
                if x2 <= x1 or y2 <= y1:
                    continue                          # fully outside the frame
                area = (x2 - x1) * (y2 - y1) / A
                if area < args.min_area or area > args.max_area:
                    bad = True
                    break
                lines.append(f"0 {((x1+x2)/2)/ww:.6f} {((y1+y2)/2)/hh:.6f} "
                             f"{(x2-x1)/ww:.6f} {(y2-y1)/hh:.6f}")
            if bad or not lines:
                dropped_area += 1
                continue
            split = tld.split_for(f"openlogo/{si}/{seen}", args.val_frac, args.test_frac)
            di, dl = out / "images" / split, out / "labels" / split
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            stem = f"ol_{si:02d}_{seen:06d}"
            cv2.imwrite(str(di / f"{stem}.jpg"), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            (dl / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1
            boxes_written += len(lines)
        if args.delete_parquet:
            shard.unlink()
            print(f"  [{si}/{len(shards)}] deleted {shard.name}")
        print(f"  [{si}/{len(shards)}] {counts}  boxes={boxes_written}", flush=True)
        if args.limit and seen >= args.limit:
            break

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n\n"
        f"nc: 1\nnames: ['logo']\n"
    )
    print(f"\nexported {sum(counts.values())} images {counts}, {boxes_written} boxes")
    print(f"dropped: {dropped_contam} benchmark-contaminated, "
          f"{dropped_cap} over density cap, {dropped_area} with an out-of-range box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
