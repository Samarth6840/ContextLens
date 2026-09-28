"""Assemble the detector training mix. Builds data, trains nothing.

Three sources, all human:
  - the reviewed frames in benchmark/eval_video/labels.jsonl (A/B/C), which carry
    both positive logo boxes and full-frame negatives
  - the balanced hard-negative crops from scripts/mine_hard_negatives.py
  - nothing else

The frozen test set is not in this script's vocabulary. It is not copied, not
counted, and is asserted absent at the end. It lives in benchmark/frozen_test/
and stays outside the training root by construction.

VAL is a frame-level carve out of A/B/C. That is within-domain validation, not
unseen-video validation: adjacent frames of one shot can land on both sides.
It is for picking a checkpoint and a threshold, and the report says so. Only the
frozen test set speaks to unseen-video generalization.
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
FROZEN_VIDEO = "88b0502222"      # the frozen test video; must appear nowhere below


def size_bucket(w: float, h: float, frame_area: float) -> str:
    a = w * h / max(frame_area, 1.0)
    return "tiny" if a < 0.005 else "small" if a < 0.02 else "medium" if a < 0.1 else "large"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="benchmark/eval_video/labels.jsonl")
    ap.add_argument("--frames", default="benchmark/eval_video/images")
    ap.add_argument("--hard-negatives", default="benchmark/hard_negatives")
    ap.add_argument("--out", default="benchmark/train_mix")
    ap.add_argument("--train-videos", default="63fb2ac4d1,7864fa1c52",
                    help="videos whose labels AND mined negatives go to TRAIN")
    ap.add_argument("--val-video", default="df01d5a843",
                    help="single video held out whole for VAL. Frame-level "
                         "carving is not used: a VAL frame sharing a scene with "
                         "a TRAIN frame measures nothing.")
    ap.add_argument("--stress-max-frac", type=float, default=0.01,
                    help="a VAL box under this share of frame area is a stress "
                         "case. 0.01 = tiny only; widen to 0.05 to sweep small in too")
    ap.add_argument("--neg-per-positive", type=float, default=1.5,
                    help="cap on empty-label negatives per positive box in TRAIN. "
                         "The cap falls on the mined crops; human full-frame "
                         "negatives are never dropped, so the achieved ratio can "
                         "exceed this. 0 = humans only, no mined crops. The "
                         "uncapped build measured 3.66 and trained to nothing.")
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()

    train_vids = {v for v in args.train_videos.split(",") if v}
    val_vid = args.val_video
    if val_vid in train_vids:
        sys.exit(f"FATAL: {val_vid} is both TRAIN and VAL")

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in Path(args.labels).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["verdict"] != "skip"]     # SKIP is not ground truth
    src = {p.name: p for p in Path(args.frames).rglob("*.jpg") if p.is_file()}

    if any(r["video"] == FROZEN_VIDEO for r in rows):
        sys.exit(f"FATAL: frozen test video {FROZEN_VIDEO} is inside the label set")

    # ── Whole-video split. VAL is one video, held out entirely, so no scene
    #    appears on both sides. The earlier frame-level carve put VAL positives
    #    in the same video as 97% of TRAIN positives, which validated nothing.
    #    Nothing is carved and nothing is shuffled: the split is the video.
    train_rows = [r for r in rows if r["video"] in train_vids]
    c_rows = [r for r in rows if r["video"] == val_vid]
    # Stress frames are CARVED OUT of VAL, not added to it. Two reasons:
    # a tiny-heavy val split pulls the selected threshold toward tiny cases and
    # off the normal distribution, and a frame that both selects a checkpoint and
    # scores it is no longer an independent measurement. Keeping them in their own
    # directory that data.yaml never points at preserves both.
    stress_rows, val_rows = [], []
    for r in c_rows:
        img = src.get(r["file"])
        if img is None or not img.is_file() or not r["boxes"]:
            val_rows.append(r)
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        tiny = any(b["w"] * b["h"] / (w * h) < args.stress_max_frac for b in r["boxes"])
        (stress_rows if tiny else val_rows).append(r)
    if not val_rows:
        sys.exit(f"FATAL: no non-skip labels for VAL video {val_vid}")
    if not train_rows:
        sys.exit(f"FATAL: no non-skip labels for TRAIN videos {args.train_videos}")

    out = Path(args.out)
    # Clear the split dirs first. Reusing a populated directory silently mixes
    # runs: 210 crops from a pre-split mine that included the VAL video were
    # still sitting in images/train, so C leaked into TRAIN while the manifest
    # happily reported by_video without them. Recreate from scratch every time.
    for d in ("images/train", "labels/train", "images/val", "labels/val",
              "images/stress", "labels/stress"):
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True, exist_ok=True)

    def write_row(r, split):
        p = src.get(r["file"])
        if p is None or not p.is_file():
            print(f"  WARN missing frame, skipped: {r['file']}")
            return None
        cv2.imwrite(str(out / "images" / split / r["file"]), cv2.imread(str(p)))
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lines = []
        for b in r["boxes"]:
            x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
            if bw < 1 or bh < 1:
                continue
            lines.append(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        (out / "labels" / split / f"{Path(r['file']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        return {"video": r["video"], "boxes": len(lines), "wh": [w, h],
                "sizes": [size_bucket(b["w"], b["h"], w * h) for b in r["boxes"] if b["w"] >= 1]}

    tr_stats, va_stats, st_stats = [], [], []
    for r in train_rows:
        s = write_row(r, "train")
        if s:
            tr_stats.append(s)
    for r in val_rows:
        s = write_row(r, "val")
        if s:
            va_stats.append(s)
    for r in stress_rows:
        s = write_row(r, "stress")
        if s:
            st_stats.append(s)

    # ── hard negatives join TRAIN as empty-label crops, CAPPED against the
    #    positives. Uncapped this mix shipped 390 crops and 64 human negative
    #    frames against 124 boxes (3.66:1), and the fine-tune that consumed it
    #    ended below its own pretrained init: val mAP50 fell 0.0015 -> 0.00006
    #    and val cls_loss climbed 5.07 -> 17.9 by epoch 5. At that ratio the
    #    cheapest prediction is "emit nothing", which is what it learned.
    #    The cap falls on the mined crops because they are the droppable part:
    #    the human full-frame negatives are 64 frames of reviewed ground truth
    #    and the crops are 390 machine crops of a single 10-minute video.
    pos_boxes = sum(s["boxes"] for s in tr_stats)
    neg_frames = sum(1 for s in tr_stats if not s["boxes"])
    budget = max(int(round(args.neg_per_positive * pos_boxes)) - neg_frames, 0)
    crops = sorted(Path(args.hard_negatives).glob("images/*.jpg"))
    rng.shuffle(crops)
    hn, dropped = 0, 0
    for img in crops:
        if hn >= budget:
            dropped += 1
            continue
        shutil.copy2(img, out / "images/train" / img.name)
        lab = Path(args.hard_negatives) / "labels" / f"{img.stem}.txt"
        (out / "labels/train" / f"{img.stem}.txt").write_text(
            lab.read_text() if lab.exists() else "")
        hn += 1

    def summarise(stats, neg_crops):
        pos_frames = [s for s in stats if s["boxes"]]
        sizes = Counter(x for s in stats for x in s["sizes"])
        return {
            "frames": len(stats) + neg_crops,
            "reviewed_frames": len(stats),
            "hard_negative_crops": neg_crops,
            "positive_frames": len(pos_frames),
            "positive_boxes": sum(s["boxes"] for s in stats),
            "negative_frames_empty_label": len(stats) - len(pos_frames),
            "by_video": dict(Counter(s["video"] for s in stats)),
            "positive_boxes_by_video": dict(Counter(
                {v: sum(s["boxes"] for s in stats if s["video"] == v)
                 for v in {s["video"] for s in stats}})),
            "box_size": dict(sizes),
        }

    manifest = {
        "seed": args.seed,
        "train_videos": sorted(train_vids), "val_video": val_vid,
        "protocol": {
            "train": f"whole videos {sorted(train_vids)} + their mined hard negatives",
            "val": f"whole video {val_vid} minus tiny cases (unseen-video)",
            "stress": f"{val_vid} frames with a box under {args.stress_max_frac:.0%} "
                      f"of frame. Measurement ONLY - data.yaml does not reference "
                      f"it, so it can never influence checkpoint or threshold choice.",
            "test": f"FROZEN, outside this root: benchmark/frozen_test/images ({FROZEN_VIDEO})",
        },
        "train": summarise(tr_stats, hn),
        "val": summarise(va_stats, 0),
        "stress": summarise(st_stats, 0),
        "frozen_test": {
            "video": FROZEN_VIDEO,
            "location": "benchmark/frozen_test/images",
            "frames": len(list((ROOT / "benchmark/frozen_test/images").glob("*.jpg"))),
            "labels": 0,
        },
    }
    tp = manifest["train"]["positive_boxes"]
    manifest["ratio_negative_to_positive"] = round(
        (manifest["train"]["negative_frames_empty_label"] + hn) / max(tp, 1), 2)
    manifest["neg_per_positive_cap"] = args.neg_per_positive
    manifest["hard_negative_crops_dropped"] = dropped
    # The cap fixes the ratio. It cannot fix single-source positives, and that is
    # the residual risk: no ratio makes 124 boxes from one video a generalizable
    # logo detector. Recorded here so no downstream report can quietly omit it.
    pos_vids = sorted({s["video"] for s in tr_stats if s["boxes"]})
    manifest["train"]["videos_carrying_positives"] = pos_vids
    manifest["train"]["positives_from_one_video"] = len(pos_vids) < 2

    # stress.yaml exists so the stress set is runnable, and is deliberately NOT
    # referenced by data.yaml.
    (out / "stress.yaml").write_text(
        f"path: {out.resolve()}\n"
        "# stress only: tiny cases carved out of VAL. Measure, never select.\n"
        f"train: images/stress\nval: images/stress\nnc: 1\nnames: ['logo']\n")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "# 1 human-detector class. Positives are human boxes; negatives are human\n"
        "# logo-free frames plus mined hard-negative crops with empty labels.\n"
        f"# val is the whole video {val_vid}, held out: unseen-video validation.\n"
        f"train: images/train\nval: images/val\nnc: 1\nnames: ['logo']\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── the assertion that matters
    hn_files = {p.name for p in Path(args.hard_negatives).glob("images/*.jpg")}
    leaks = {
        "train": sorted(f for f in (p.name for p in (out / "images/train").glob("*.jpg"))
                        if f.startswith(FROZEN_VIDEO)),
        "val": sorted(f for f in (p.name for p in (out / "images/val").glob("*.jpg"))
                      if f.startswith(FROZEN_VIDEO)),
        "stress": sorted(f for f in (p.name for p in (out / "images/stress").glob("*.jpg"))
                         if f.startswith(FROZEN_VIDEO)),
        "hard_negatives": sorted(f for f in hn_files if f.startswith(FROZEN_VIDEO)),
    }
    manifest["frozen_test_leak_check"] = {k: len(v) for k, v in leaks.items()}
    manifest["frozen_test_leaked"] = {k: v for k, v in leaks.items() if v}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Assert against the FILES ON DISK, not against the row lists used to build
    # them. The row lists were correct while the directory was not, which is
    # precisely the failure this check exists to catch.
    for split in ("train", "val", "stress"):
        names = [p.name for p in (out / "images" / split).glob("*.jpg")]
        strays = sorted({n.split("_")[0] for n in names}
                        - (train_vids | {val_vid, FROZEN_VIDEO}))
        if strays:
            sys.exit(f"FATAL: {split}/ contains unknown videos {strays}")
        if split == "train" and val_vid in {n.split("_")[0] for n in names}:
            sys.exit(f"FATAL: VAL video {val_vid} present in train/")
        if FROZEN_VIDEO in {n.split("_")[0] for n in names}:
            sys.exit(f"FATAL: frozen TEST video present in {split}/")
    manifest["verified_on_disk"] = {
        s: dict(Counter(p.name.split("_")[0] for p in (out / "images" / s).glob("*.jpg")))
        for s in ("train", "val", "stress")}

    if any(leaks.values()):
        sys.exit(f"FATAL: frozen test video leaked: {manifest['frozen_test_leaked']}")

    print(json.dumps(manifest, indent=2))
    print(f"\n-> {out}   (no training performed)")
    if manifest["train"]["positives_from_one_video"]:
        print(f"\nWARNING: every TRAIN positive box comes from {pos_vids} "
              f"({tp} boxes). The negative cap cannot fix that; label another "
              f"video's logos or do not read VAL as generalization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
