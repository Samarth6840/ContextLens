"""Mine true logo-FREE crops from LogoDet-3K to serve as detector negatives.

YOLO trains every unlabelled region as background, so a detector trained only on
logo-saturated frames never sees a negative example and over-fires. The
LogoDet-3K corpus has no logo-free images, but it does have large regions that
carry no annotation at all. This crops the largest fully-unannotated window from
each image and writes it with an EMPTY label file, which is a real negative.

The mask is built from the COMPLETE annotation in the parquet (median 161
boxes/image, up to 1717), not from the exported label file. Building the mask
from capped labels would mark unexported logos as background and produce
mislabeled negatives.

The test split is excluded so the held-out benchmark stays untouched.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "tld", Path(__file__).resolve().parent / "train_logo_detector.py"
)
_tld = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tld)


def full_annotation(parquet_dir: Path, keep_splits: set[str]) -> dict:
    """path -> (all boxes, decoded image). Only images in keep_splits."""
    boxes: dict[str, list] = defaultdict(list)
    raw: dict[str, bytes] = {}
    seen: set[str] = set()
    for shard in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["image_path", "bbox"])
        for img, bbox in zip(df["image_path"], df["bbox"]):
            path = img.get("path") if isinstance(img, dict) else str(img)
            if path in seen:
                continue
            if _tld.split_for(path, 0.15, 0.25) not in keep_splits:
                continue
            blob = img.get("bytes") if isinstance(img, dict) else None
            if not isinstance(blob, (bytes, bytearray)):
                continue
            seen.add(path)
            raw[path] = bytes(blob)
            if bbox is not None and len(bbox) >= 4:
                boxes[path] = [
                    tuple(float(v) for v in np.asarray(bbox).ravel()[:4])
                ]
    # Second pass for remaining boxes of each image (one box per row).
    for shard in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(shard, columns=["image_path", "bbox"])
        for img, bbox in zip(df["image_path"], df["bbox"]):
            path = img.get("path") if isinstance(img, dict) else str(img)
            if path not in raw or bbox is None or len(bbox) < 4:
                continue
            boxes[path].append(
                tuple(float(v) for v in np.asarray(bbox).ravel()[:4])
            )
    out = {}
    for path, blob in raw.items():
        arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            out[path] = (arr, boxes.get(path, []))
    return out


def clean_windows(arr: np.ndarray, boxes: list, frac: float, min_px: int,
                  max_per_image: int):
    """Fully-unannotated axis-aligned windows of `frac` x image size.

    Scans a k x k grid of candidate windows and keeps the free ones, so one
    logo-saturated frame can yield several negatives. A fully outside box
    contributes no mask, so overhanging annotations are ignored. Returns a
    list of (x1, y1, x2, y2).
    """
    h, w = arr.shape[:2]
    wh, ww = int(h * frac), int(w * frac)
    if wh < min_px or ww < min_px:
        return []
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    free = (1 - mask).astype(np.uint8)
    ii = cv2.integral(free)
    k = max(1, int(np.floor(min(h / wh, w / ww))))
    ys = np.linspace(0, h - wh, k).astype(int) if k > 1 else [0]
    xs = np.linspace(0, w - ww, k).astype(int) if k > 1 else [0]
    out: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            n = int(ii[y + wh, x + ww] - ii[y, x + ww] - ii[y + wh, x] + ii[y, x])
            if n == wh * ww:
                out.append((int(x), int(y), int(x + ww), int(y + wh)))
                if len(out) >= max_per_image:
                    return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/tmp/realbench/data")
    ap.add_argument("--root", default="weights/logo_detector")
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--min-px", type=int, default=64)
    ap.add_argument("--max-per-image", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    counts: dict[str, int] = defaultdict(int)
    made: dict[str, int] = defaultdict(int)
    for path, (arr, boxes) in sorted(full_annotation(Path(args.parquet), {"train", "val"}).items()):
        wins = clean_windows(arr, boxes, args.frac, args.min_px, args.max_per_image)
        if not wins:
            continue
        split = _tld.split_for(path, 0.15, 0.25)
        counts[split] += len(wins)
        if args.dry_run:
            continue
        di = root / "negatives" / split
        dl = root / "labels_neg" / split
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        for i, (x1, y1, x2, y2) in enumerate(wins):
            name = f"neg_{Path(path).stem}_{i}.jpg"
            if (di / name).exists():
                made[split] += 1
                continue
            cv2.imwrite(str(di / name), arr[y1:y2, x1:x2],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            (dl / f"neg_{Path(path).stem}_{i}.txt").write_text("")
            made[split] += 1

    found = dict(counts)
    if args.dry_run:
        print(f"clean {args.frac:.0%} window available: {found}")
        print(f"would write (test excluded): {dict(made)}")
    else:
        print(f"clean {args.frac:.0%} window available: {found}")
        print(f"wrote negatives: {dict(made)}")
        _wire_data_yaml(root)
    return 0


def _wire_data_yaml(root: Path) -> None:
    """Point data.yaml at negatives/ so Ultralytics actually loads them.

    Ultralytics only looks for label files under images/, so negatives written
    there are silently ignored. Rewrites train/val as two-path lists.
    """
    data = root / "data.yaml"
    if not data.exists():
        return
    negs = {s: len(list((root / "negatives" / s).glob("*.jpg"))) for s in ("train", "val")}
    if not any(negs.values()):
        return
    data.write_text(
        f"path: {root.resolve()}\n"
        f"# mined logo-free negatives (empty label files)\n"
        f"train: [images/train, negatives/train]\n"
        f"val: [images/val, negatives/val]\n\n"
        f"nc: 1\nnames: ['logo']\n"
    )
    print(f"data.yaml now includes {negs['train']} train / {negs['val']} val negatives")


if __name__ == "__main__":
    sys.exit(main())
