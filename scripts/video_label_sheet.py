"""Contact sheet for labelling ContextLens video frames.

The extracted video frames are a candidate pool with no ground truth, and we
do not have enough distinct videos to label all of them by hand. A numbered
grid is far faster to review than individual files: open the sheet, tick the
frames that are genuinely logo-free, and that list comes straight back as a
YOLO negatives directory.

Three rules make a returned index list actually mean something:

1. The sheet is filled ROUND-ROBIN across source videos, and capped per video
   (`--per-video`). A sheet that is 48 tiles of one video cannot be judged -
   it looks like 48 independent samples but they are one scene's worth of
   evidence, and a label set drawn from it is not generalisable.
2. Every tile is labelled `<video_md5>#<source_frame_index>`, and the sidecar
   JSON carries `video` and `frame_index`. A reviewer can say which shot a
   verdict refers to, and a returned index resolves to exactly one file.
3. `--apply "<indices>"` turns that list into a labelled YOLO split and
   prints the video mix. Without it the answer has no destination.

    python scripts/video_label_sheet.py --frames benchmark/eval_video/images/train \
        --out benchmark/eval_video/review/sheet_train.png \
        --weights weights/logo_detector/train/weights/best.pt
    python scripts/video_label_sheet.py --frames ... --out ... \
        --apply "3 7 12" --dataset benchmark/eval_video/labelled
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def video_of(stem: str) -> str:
    """`<video_md5>_<source_frame_index>` -> video id. The index is the identity."""
    return stem.rsplit("_", 1)[0]


def frame_index_of(stem: str) -> int:
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def pick_frames(frames: list[Path], sheet: int, per_video: int) -> list[Path]:
    """Round-robin across videos, evenly spread within each, so no video floods.

    Within a video the frames are already a fixed stride over the whole clip, so
    an even subsample of them is an even subsample of the video's timeline.
    """
    by_video: dict[str, list[Path]] = defaultdict(list)
    for f in frames:
        by_video[video_of(f.stem)].append(f)
    picks: list[Path] = []
    order = sorted(by_video)
    for v in order:
        pool = by_video[v]
        take = min(per_video, len(pool))
        if sheet:
            take = min(take, max(1, sheet - len(picks)))
        if take >= len(pool):
            chosen = pool
        elif take == 1:
            chosen = pool[len(pool) // 2:]
        else:
            idx = [round(i * (len(pool) - 1) / (take - 1)) for i in range(take)]
            chosen = [pool[i] for i in sorted(set(idx))]
        picks.extend(chosen)
        if sheet and len(picks) >= sheet:
            break
    return picks[:sheet] if sheet else picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of frames")
    ap.add_argument("--out", help="output png (skipped when --apply is used)")
    ap.add_argument("--weights", default="", help="draw detections from these weights")
    ap.add_argument("--sheet", type=int, default=48, help="max frames on the sheet")
    ap.add_argument("--per-video", type=int, default=16,
                    help="max tiles from any single video")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. mps, 0, cpu")
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--apply", default="",
                    help="reviewed indices, e.g. '3 7 12' -> labelled YOLO split")
    ap.add_argument("--dataset", default="",
                    help="output dir for --apply (default: alongside --frames)")
    args = ap.parse_args()

    # Recursive so one command can cover a whole pool split across per-split
    # subdirs - otherwise the reviewer gets a sheet per video again, which is
    # the problem this script exists to remove.
    frames = sorted(p for p in Path(args.frames).rglob("*.jpg") if p.is_file())
    if not frames:
        sys.exit(f"FATAL: no frames under {args.frames}")
    if not args.out and not args.apply:
        sys.exit("FATAL: pass --out (draw a sheet) or --apply (consume a review)")

    picked = pick_frames(frames, args.sheet, args.per_video)
    root = Path(args.frames)
    index_path = None
    if args.out:
        index_path = Path(args.out).with_suffix(".json")
        if index_path.exists():
            known = {e["file"] for e in json.loads(index_path.read_text())}
            picked = [f for f in picked if str(f.relative_to(root)) in known]

    predictor = None
    if args.weights:
        from ultralytics import YOLO
        predictor = YOLO(args.weights)

    cols = args.cols
    rows = (len(picked) + cols - 1) // cols
    cell, bar = args.cell, 22
    sheet_img = 255 * np.ones((rows * (cell + bar), cols * cell, 3), np.uint8)

    index = []
    for i, f in enumerate(picked):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  WARN unreadable, skipped: {f.name}")
            continue
        boxes = []
        if predictor is not None:
            r = predictor.predict(img, imgsz=args.imgsz, conf=args.conf,
                                  device=args.device, verbose=False)[0]
            boxes = [[int(v) for v in xy] for xy, _ in
                     zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        h, w = img.shape[:2]
        s = (cell - 4) / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        r_, c_ = divmod(i, cols)
        y0 = r_ * (cell + bar) + bar
        x0 = c_ * cell
        sheet_img[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        vid = video_of(f.stem)
        cv2.putText(sheet_img, f"[{i}] {vid}#{frame_index_of(f.stem)}  d{len(boxes)}",
                    (x0 + 3, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 160), 1, cv2.LINE_AA)
        index.append({"i": i, "file": str(f.relative_to(root)),
                      "video": vid, "frame_index": frame_index_of(f.stem),
                      "detections": len(boxes), "boxes": boxes})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, sheet_img)
        Path(args.out).with_suffix(".json").write_text(json.dumps(index, indent=1))
        mix = defaultdict(int)
        for e in index:
            mix[e["video"]] += 1
        print(f"{len(index)} frames -> {args.out} ({cols}x{rows})")
        print(f"  videos on sheet: {dict(mix)}")
        print(f"  map -> {Path(args.out).with_suffix('.json')}")
        if not any(e["detections"] for e in index):
            print("  NOTE: no --weights given, so no boxes are drawn. A sheet with no")
            print("        boxes cannot be checked against the detector - re-run with")
            print("        --weights before reviewing.")

    if args.apply:
        apply_review(index, {int(x) for x in args.apply.split()},
                     Path(args.dataset) if args.dataset else Path(args.frames).parent / "labelled",
                     Path(args.frames))
    elif not args.weights:
        print("\nReview and reply with the indices that are LOGO-FREE, e.g. "
              "'negatives: 3 7 12 19 24'.")
    return 0


def apply_review(index: list[dict], wanted: set[int], out: Path,
                 frames_dir: Path) -> int:
    """Write the reviewed LOGO-FREE frames as a YOLO negatives split.

    Empty label file = YOLO background. The reviewer's verdict is the only
    ground truth in the loop, so the frames are copied verbatim: no crop, no
    rescale, nothing that could turn a verified negative into a different one.
    """
    by_i = {e["i"]: e for e in index}
    missing = sorted(wanted - set(by_i))
    if missing:
        sys.exit(f"FATAL: indices not on this sheet: {missing}")
    imgs, labs = out / "images", out / "labels"
    imgs.mkdir(parents=True, exist_ok=True)
    labs.mkdir(parents=True, exist_ok=True)
    mix = defaultdict(int)
    for i in sorted(wanted):
        e = by_i[i]
        src = frames_dir / e["file"]
        if not src.exists():
            sys.exit(f"FATAL: {src} gone")
        name = f"{e['video']}_{e['frame_index']:06d}.jpg"
        cv2.imwrite(str(imgs / name), cv2.imread(str(src)),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        (labs / f"{name[:-4]}.txt").write_text("")
        mix[e["video"]] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n# human-verified logo-free frames\n"
        f"train: images\nval: images\n\nnc: 1\nnames: ['logo']\n"
    )
    print(f"\napplied {len(wanted)} reviewed negatives -> {out}")
    print(f"  per video: {dict(mix)}")
    if len(mix) < 2:
        print("  WARNING: all negatives come from ONE video. That is one scene's")
        print("           worth of evidence - fine to sanity-check the detector,")
        print("           not enough to train or benchmark on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
