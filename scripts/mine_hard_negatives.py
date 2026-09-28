"""Mine hard negatives: the boxes the detector fired on that are not logos.

The audit says the detector learned "something visually interesting = logo"
rather than "this region is actually a logo". Recall is 0.98 and it fires on
88.6% of logo-free frames, so the training signal it needs is not more logos --
it is a large set of confidently-wrong boxes on real ad frames.

Three rules this script exists to enforce:

1. Only TRAIN videos. VAL and TEST frames are never opened. The frozen test set
   lives outside benchmark/eval_video/images/ precisely so it cannot be reached
   here by path accident; --train-only is the second lock on the same door.
2. Sample, do not dump. 1027 FPs against 129 real logo boxes would teach
   "almost nothing is a logo". The pool is balanced across video, scene, box
   size, aspect and screen position, and trimmed to a target count.
3. Keep the surrounding context. A tight crop teaches "this texture is not a
   logo". A crop with context around it teaches the boundary -- where a logo
   ends and a UI button or an ad's decorative type begins -- which is the
   distinction actually failing. Boxes are padded by a fraction of their own
   size, so a large graphic gets a wide view and a small chip a tight one.

A negative crop carries an EMPTY label file. That is what makes it a negative.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_detector_separation import iou_match  # noqa: E402

# Proxies for "visual type", since nothing here may consult a classifier and a
# semantic label would be exactly the shortcut the model already learned.
def size_bucket(w: float, h: float, frame_area: float) -> str:
    a = w * h / max(frame_area, 1.0)
    return "tiny" if a < 0.005 else "small" if a < 0.02 else "medium" if a < 0.1 else "large"


def aspect_bucket(w: float, h: float) -> str:
    r = w / max(h, 1e-6)
    return "wide" if r > 2.0 else "tall" if r < 0.5 else "squareish"


def pos_bucket(cx: float, cy: float) -> str:
    return "centre" if 0.33 < cx < 0.67 and 0.33 < cy < 0.67 else "edge"


def scene_of(video: str, frame_index: int) -> str:
    """Coarse scene id so sampling spreads across shots, not across one shot."""
    return f"{video}:{frame_index // 50}"


def balanced_sample(items: list[dict], target: int) -> list[dict]:
    """Round-robin over the rarest stratum first, so no single video, scene,
    size, aspect or position dominates just because it produced more boxes."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        strata[(it["video"], it["size"], it["aspect"], it["pos"])].append(it)
    for v in strata.values():
        v.sort(key=lambda d: d["conf"])          # keep the model's best guesses
    order = sorted(strata, key=lambda k: (len(strata[k]), k))
    out, i = [], 0
    while len(out) < target and any(strata[k] for k in order):
        k = order[i % len(order)]
        if strata[k]:
            out.append(strata[k].pop())
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", default="benchmark/eval_video/images")
    ap.add_argument("--labels", default="benchmark/eval_video/labels.jsonl")
    ap.add_argument("--out", default="benchmark/hard_negatives")
    ap.add_argument("--target", type=int, default=390, help="negatives to keep (3:1 vs 129)")
    ap.add_argument("--conf", type=float, default=0.05, help="the audit's best-F1 point")
    ap.add_argument("--videos", default="",
                    help="comma-separated video md5s to mine. Defaults to every "
                         "video under --frames. Must EXCLUDE any video used as "
                         "VAL: mining a VAL video's false positives puts that "
                         "video's failures into TRAIN, which is leakage.")
    ap.add_argument("--pad", type=float, default=0.5, help="context pad, fraction of box size")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    frames_root = Path(args.frames)
    rows = {}
    for line in Path(args.labels).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["verdict"] != "skip":
                rows[r["file"]] = r

    # Guard 1: never open a frame outside the TRAIN pool.
    frozen = ROOT / "benchmark/frozen_test/images"
    train_files = sorted(p for p in frames_root.rglob("*.jpg") if p.is_file())
    leaked = [p for p in train_files if frozen in p.parents]
    if leaked:
        sys.exit(f"FATAL: {len(leaked)} frozen-TEST frames inside {frames_root}")
    if args.videos:
        keep = set(args.videos.split(","))
        train_files = [p for p in train_files if p.name.split("_")[0] in keep]
        if not train_files:
            sys.exit(f"FATAL: no frames match --videos {args.videos}")
    print(f"TRAIN pool: {len(train_files)} frames, frozen TEST excluded"
          + (f", restricted to {args.videos}" if args.videos else ""))

    from ultralytics import YOLO
    model = YOLO(args.model)

    cands: list[dict] = []
    seen = Counter()
    for p in train_files:
        r = rows.get(p.name)
        gts = np.array([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
                        for b in (r["boxes"] if r else [])], dtype=float).reshape(-1, 4)
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        b = res.boxes
        xy = b.xyxy.cpu().numpy() if b is not None and len(b) else np.zeros((0, 4))
        cf = b.conf.cpu().numpy() if b is not None and len(b) else np.zeros((0,))
        if not len(xy):
            continue
        dets = np.hstack([xy, cf[:, None]])
        tp, n = iou_match(dets, gts, args.iou)
        matched = _matched_mask(dets, gts, args.iou)
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        seen[p.name] = n
        for di in range(n):
            if matched[di]:
                continue
            x1, y1, x2, y2, cf_ = dets[di]
            bw, bh = x2 - x1, y2 - y1
            cands.append({
                "file": p.name, "path": str(p), "video": r["video"] if r else p.name.split("_")[0],
                "frame_index": r["frame_index"] if r else -1,
                "conf": round(float(cf_), 3),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "size": size_bucket(bw, bh, w * h),
                "aspect": aspect_bucket(bw, bh),
                "pos": pos_bucket((x1 + x2) / 2 / w, (y1 + y2) / 2 / h),
                "scene": scene_of(p.name.split("_")[0], r["frame_index"] if r else 0),
                "frame_wh": [w, h],
            })
        if len(cands) and len(seen) % 50 == 0:
            print(f"  {len(seen)} frames, {len(cands)} false positives", flush=True)

    print(f"\n{len(cands)} false positives mined from TRAIN")
    if not cands:
        sys.exit("FATAL: no false positives found; nothing to mine")

    keep = balanced_sample(cands, args.target)
    out = Path(args.out)
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for f in (out / "images").glob("*.jpg"):
        f.unlink()
    for f in (out / "labels").glob("*.txt"):
        f.unlink()

    for i, c in enumerate(keep):
        img = cv2.imread(c["path"])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = c["box"]
        bw, bh = x2 - x1, y2 - y1
        # Proportional context: a wide pad on a big graphic, a tight one on a
        # small chip. Clipped to the frame, and never allowed to reach outside it.
        px, py = bw * args.pad, bh * args.pad
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        name = f"{c['video']}_{c['frame_index']:06d}_{i:04d}.jpg"
        cv2.imwrite(str(out / "images" / name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        (out / "labels" / f"{Path(name).stem}.txt").write_text("")   # empty == negative
        c["crop"] = name
        c["crop_box"] = [cx1, cy1, cx2, cy2]
        c["orig_box_rel"] = [(x1 - cx1) / (cx2 - cx1), (y1 - cy1) / (cy2 - cy1),
                             bw / (cx2 - cx1), bh / (cy2 - cy1)]

    manifest = {
        "model": args.model, "conf": args.conf, "iou": args.iou,
        "pad": args.pad, "target": args.target,
        "source_frames": len(train_files), "mined_false_positives": len(cands),
        "kept": len(keep),
        "balance_by_video": dict(Counter(c["video"] for c in keep)),
        "balance_by_size": dict(Counter(c["size"] for c in keep)),
        "balance_by_aspect": dict(Counter(c["aspect"] for c in keep)),
        "balance_by_position": dict(Counter(c["pos"] for c in keep)),
        "scenes_covered": len({c["scene"] for c in keep}),
        "frozen_test_touched": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "negatives.json").write_text(json.dumps(keep, indent=1))
    print(json.dumps(manifest, indent=2))
    print(f"\n-> {out}   ({len(keep)} empty-label crops)")
    print("positives come from the human export, not from here: "
          "benchmark/video_labelled has the 129 real logo boxes.")
    return 0


def _matched_mask(dets, gts, thr):
    """Which detections are true positives, so the rest can be kept as negatives."""
    if len(gts) == 0:
        return np.zeros(len(dets), bool)
    if len(dets) == 0:
        return np.zeros(0, bool)
    mask = np.zeros(len(dets), bool)
    used = set()
    for di in np.argsort(-dets[:, 4]):
        best, bj = thr, -1
        for gj, g in enumerate(gts):
            if gj in used:
                continue
            ix1, iy1 = max(dets[di, 0], g[0]), max(dets[di, 1], g[1])
            ix2, iy2 = min(dets[di, 2], g[2]), min(dets[di, 3], g[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue
            ua = ((dets[di, 2] - dets[di, 0]) * (dets[di, 3] - dets[di, 1])
                  + (g[2] - g[0]) * (g[3] - g[1]) - inter)
            v = inter / ua if ua > 0 else 0.0
            if v >= thr and v > best:
                best, bj = v, gj
        if bj >= 0:
            used.add(bj)
            mask[di] = True
    return mask


if __name__ == "__main__":
    sys.exit(main())
