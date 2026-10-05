"""Quantify the domain gap between public logo photos and ContextLens video frames.

Do not train on a public logo dataset on the assumption that "more logos is
better". A detector inherits the statistics of what it was shown, and the
statistics of LogoDet-3K are not the statistics of a phone-shot video frame:

  * LogoDet-3K is sharp, high-resolution product/street photography.
  * ContextLens frames are 360p-720p, H.264-compressed, motion-blurred, and a
    logo is usually a small mark inside a large cluttered frame.

This measures the axes that decide whether a transfer is even plausible: object
size, density, aspect ratio, sharpness, and frame-covering rate. It also audits
the training set against itself, because a label that covers the whole frame is
not a logo and the detector is then trained to fire on scenes.

The two sides are NOT measured the same way, and the report says so on every
line: LogoDet numbers come from human GT boxes, video numbers come from the
detector's own predictions, because the video pool has no reviewed labels yet.
Comparing a GT distribution to a predicted distribution is weaker evidence than
comparing two GT distributions - the video column is an upper bound on what a
model trained on LogoDet-style data can expect, not a measurement of the truth.

    python scripts/domain_mismatch.py --video-weights weights/logo_detector/train/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RESIZE = 512  # fixed long side: Laplacian variance is scale-dependent, so every
              # image is normalised to the same size before it is compared.


def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. 0 = flat, high = detailed/sharp."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def load_resized(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = RESIZE / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    return img


def measure(boxes_per_image: list[list[tuple[float, float, float, float]]],
            images: list[np.ndarray]) -> dict:
    """boxes: [(x1,y1,x2,y2)] in RESIZED-image pixel coords."""
    n_img = len(images)
    areas, ars, box_sharp, frame_sharp = [], [], [], []
    per_img = []
    for img, boxes in zip(images, boxes_per_image):
        fs = sharpness(img)
        frame_sharp.append(fs)
        H, W = img.shape[:2]
        per_img.append(len(boxes))
        for (x1, y1, x2, y2) in boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            areas.append(bw * bh / (W * H))
            ars.append(bw / bh)
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size:
                box_sharp.append(sharpness(crop))
    a = np.array(areas) if areas else np.zeros(0)
    r = np.array(ars) if ars else np.zeros(0)
    bs = np.array(box_sharp) if box_sharp else np.zeros(0)
    fs = np.array(frame_sharp) if frame_sharp else np.zeros(0)
    q = lambda x, p: (None if not len(x) else round(float(np.percentile(x, p)), 5))
    return {
        "images": n_img,
        "boxes_total": int(a.size),
        "boxes_per_image": round(float(np.mean(per_img)), 1) if per_img else 0.0,
        "boxes_per_image_median": round(float(np.median(per_img)), 1) if per_img else 0.0,
        "area_frac_median": q(a, 50), "area_frac_p10": q(a, 10), "area_frac_p25": q(a, 25),
        "area_frac_p75": q(a, 75), "area_frac_p90": q(a, 90),
        "frac_boxes_over_50pct_frame": round(float((a > 0.5).mean()), 4) if len(a) else None,
        "frac_boxes_under_1pct_frame": round(float((a < 0.01).mean()), 4) if len(a) else None,
        "aspect_median": q(r, 50), "aspect_p95": q(r, 95),
        "frac_wider_than_3to1": round(float((r > 3).mean()), 4) if len(r) else None,
        "frame_sharpness_median": q(fs, 50), "frame_sharpness_p10": q(fs, 10),
        "box_sharpness_median": q(bs, 50),
        # Scale-free: how sharp is the logo region relative to its own frame.
        # A logo that is much softer than the frame around it is a different
        # detection problem from a crisp mark on a crisp photo.
        "box_over_frame_sharpness": (
            None if not len(bs) or not len(fs) or float(np.median(fs)) <= 0
            else round(float(np.median(bs) / float(np.median(fs))), 3)),
    }


def logo_detect_side(root: Path, split: str, limit: int) -> tuple[dict, list]:
    """Human GT boxes from a YOLO-exported LogoDet root."""
    per_img, images = [], []
    labs = sorted((root / "labels" / split).glob("*.txt"))
    for lab in labs[:limit]:
        img = load_resized(root / "images" / split / f"{lab.stem}.jpg")
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for line in lab.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H))
        images.append(img)
        per_img.append(boxes)
    return measure(per_img, images), per_img


def video_side(frames_root: Path, weights: str, imgsz: int, device: str | None,
               limit: int) -> dict:
    """Detector-predicted boxes on real ContextLens frames. NOT ground truth."""
    from ultralytics import YOLO
    model = YOLO(weights)
    files = sorted(p for p in frames_root.rglob("*.jpg") if p.is_file())[:limit]
    per_img, images = [], []
    for f in files:
        img = load_resized(f)
        if img is None:
            continue
        # predict on the ORIGINAL file, then rescale boxes to RESIZE coords
        r = model.predict(str(f), imgsz=imgsz, conf=0.10, device=device,
                          verbose=False)[0]
        h0, w0 = model_predict_hw(f)
        s = (img.shape[0] / h0) if h0 else 1.0
        d = r.boxes
        xy = d.xyxy.cpu().numpy() if d is not None and len(d) else np.zeros((0, 4))
        images.append(img)
        per_img.append([tuple(b * s) for b in xy])
    m = measure(per_img, images)
    m["boxes_are"] = "DETECTOR PREDICTIONS, not reviewed ground truth"
    m["frames_measured"] = len(images)
    return m


_HW_CACHE: dict[str, tuple[int, int]] = {}


def model_predict_hw(p: Path) -> tuple[int, int]:
    key = str(p)
    if key not in _HW_CACHE:
        a = cv2.imread(str(p))
        _HW_CACHE[key] = (a.shape[0], a.shape[1]) if a is not None else (0, 0)
    return _HW_CACHE[key]


def report(logodet: dict, video: dict) -> None:
    print("\n" + "=" * 74)
    print("DOMAIN MISMATCH:  LogoDet-3K (GT boxes)  vs  ContextLens video (predicted)")
    print("=" * 74)
    print(f"{'metric':<30} {'LogoDet (GT)':>18} {'Video (pred)':>18}")
    print("-" * 74)
    rows = [
        ("images", "images", "{:>18}"),
        ("frames/photos", "frames_measured", "{:>18}"),
        ("boxes total", "boxes_total", "{:>18}"),
        ("boxes per image", "boxes_per_image", "{:>18}"),
        ("median box area / frame", "area_frac_median", "{:>18.5f}"),
        ("  p10 area / frame", "area_frac_p10", "{:>18.5f}"),
        ("  p90 area / frame", "area_frac_p90", "{:>18.5f}"),
        ("  boxes over 50% of frame", "frac_boxes_over_50pct_frame", "{:>18.4f}"),
        ("  boxes under 1% of frame", "frac_boxes_under_1pct_frame", "{:>18.4f}"),
        ("median aspect ratio", "aspect_median", "{:>18.2f}"),
        ("  wider than 3:1", "frac_wider_than_3to1", "{:>18.4f}"),
        ("median frame sharpness", "frame_sharpness_median", "{:>18.1f}"),
        ("  box/frame sharpness", "box_over_frame_sharpness", "{:>18.3f}"),
    ]
    for label, key, fmt in rows:
        a, b = logodet.get(key), video.get(key)
        sa = fmt.format(a) if isinstance(a, (int, float)) else f"{'-':>18}"
        sb = fmt.format(b) if isinstance(b, (int, float)) else f"{'-':>18}"
        print(f"{label:<30} {sa} {sb}")
    print("-" * 74)
    print("Video column = detector output, not reviewed truth. Treat it as an upper")
    print("bound on achievable box size, and re-measure once the pool is labelled.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo-root", default="weights/logo_detector")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--video-frames", default="benchmark/eval_video/images")
    ap.add_argument("--video-weights", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="benchmark/results/domain_mismatch.json")
    args = ap.parse_args()

    root = Path(args.logo_root)
    if not (root / "labels" / args.split).exists():
        sys.exit(f"FATAL: no labels at {root}/labels/{args.split}")
    logodet, _ = logo_detect_side(root, args.split, args.limit)
    print(f"LogoDet  {args.split}: {logodet['images']} images, "
          f"{logodet['boxes_total']} GT boxes")

    if not args.video_weights:
        report(logodet, {})
        print("no --video-weights: skipped the video side")
        return 0
    video = video_side(Path(args.video_frames), args.video_weights, args.imgsz,
                       args.device, args.limit)
    print(f"Video    predicted: {video['frames_measured']} frames, "
          f"{video['boxes_total']} predicted boxes")
    report(logodet, video)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "logo_det": logodet, "video": video,
        "caveat": "video boxes are detector predictions, not reviewed ground truth",
        "resized_long_side": RESIZE,
    }, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
