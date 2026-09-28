"""Measure whether the detector separates LOGO from NON-LOGO.

The brand-blind single-class detector has only ever been graded against
logo-positive images, so its precision on logo-free frames was unmeasured.
LogoDet-3K has no logo-free frames, so the val split carries mined negatives:
crops that are provably free of any annotation in the full parquet, written
with empty label files.

Reports the ledger the detector work needs: mAP50 / mAP50-95 / precision /
recall (Ultralytics val), then F1, TP/FP/FN-per-image and PR-AUC over a
confidence sweep, the headline AUROC of per-image max confidence, and two
breakdowns that a single aggregate hides:

  * per source video - one video can be perfect while the next is worthless,
    and the pooled number averages that away;
  * per object size - ContextLens frames are full of small on-screen marks, so
    tiny-logo recall is the number that decides whether this detector is usable.
"""

import argparse
import importlib.util as ilu
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

# COCO pixel-area cut-points, so the size buckets mean the same thing as any
# other detector report. `tiny` is separated from `small` on purpose: a mark
# under 32x32 px is a different detection problem, not a smaller `small`.
SIZE_EDGES_PX = {"tiny": 32, "small": 96, "medium": 384}


def size_bucket(px_area: float) -> str:
    for name, edge in SIZE_EDGES_PX.items():
        if px_area < edge * edge:
            return name
    return "large"


def group_key(stem: str) -> str:
    """`<video_md5>_<frame_index>` -> video id; anything else is its own group."""
    m = re.match(r"^([0-9a-f]{6,})_\d+(?:\.jpg)?$", stem)
    return m.group(1) if m else stem.removesuffix(".jpg")


def _load(path, name):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ev = _load(Path(__file__).resolve().parent / "eval_logo.py", "eval_logo_mod")
_auroc = _ev._auroc


def read_split(root: Path, split: str):
    """[(name, np.array of xyxy GT, is_logo_image, img_area)]; empty array = negative.

    `is_logo` is decided by the LABEL FILE, never by the directory. Two layouts
    put negatives in different places: mined crops live in negatives/<split> with
    empty labels in labels_neg/, and a hand-reviewed video set puts every frame
    in images/<split> with an EMPTY label file for a frame the reviewer asserted
    was logo-free. Reading the directory instead of the label file scores those
    real negatives as logo images with zero ground truth, which corrupts exactly
    the two numbers the video set exists to produce: AUROC, and FP per
    logo-free frame.
    """
    out = []
    splits = (["train", "val", "test"] if split == "all"
              else [split])
    for sp in splits:
        for sub, lsub in (("images", "labels"), ("negatives", "labels_neg")):
            if not (root / sub / sp).is_dir():
                continue
            for img in sorted((root / sub / sp).glob("*.jpg")):
                lab = root / lsub / sp / f"{img.stem}.txt"
                w, h = _imsize(img)
                gts = []
                if lab.exists() and lab.read_text().strip():
                    for line in lab.read_text().splitlines():
                        v = line.split()
                        if len(v) < 5:
                            continue
                        _, cx, cy, bw, bh = (float(x) for x in v[:5])
                        gts.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                    (cx + bw / 2) * w, (cy + bh / 2) * h])
                # `sub` rides along because is_logo no longer implies the
                # directory: a reviewed logo-free frame lives in images/ and must
                # still be read from there, not from negatives/.
                out.append((img.stem, np.array(gts, dtype=float) if gts
                            else np.zeros((0, 4)), bool(gts), w * h, sub, sp))
    if not out:
        sys.exit(f"FATAL: no images found under {root} for split={split!r}. "
                 f"Use --split all to grade every split at once (an audit of a "
                 f"2-video set has no honest single held-out split to grade).")
    return out


def _imsize(p: Path):
    import cv2
    a = cv2.imread(str(p))
    return (a.shape[1], a.shape[0]) if a is not None else (1, 1)


def iou_match(dets: np.ndarray, gts: np.ndarray, thr: float = 0.5):
    """Greedy IoU match. Returns (tp, n_dets) for one image."""
    if len(gts) == 0 or len(dets) == 0:
        return 0, len(dets)
    used, tp = set(), 0
    order = np.argsort(-dets[:, 4])
    for di in order:
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
            tp += 1
    return tp, len(dets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="weights/logo_detector")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    root = Path(args.root)
    model = YOLO(args.model)

    v = model.val(data=str(root / "data.yaml"), imgsz=960, device=args.device,
                  verbose=False)
    box = v.box
    if args.split == "all":
        print("\nNOTE: --split all grades every split as one pooled set. That is an")
        print("      AUDIT of a 2-video pool, not a held-out score. mAP below still")
        print("      comes from the data.yaml 'val:' key only.")
    print(f"\n=== detection quality ({args.split}) ===")
    print(f"  mAP50    {box.map50:.4f}")
    print(f"  mAP50-95 {box.map:.4f}")
    print(f"  precision{box.mp:.4f}   recall {box.mr:.4f}")

    items = read_split(root, args.split)
    rows = []
    for stem, gts, is_logo, img_area, sub, sp in items:
        img = root / sub / sp / f"{stem}.jpg"
        if not img.exists():
            continue
        r = model.predict(str(img), imgsz=960, conf=0.001, device=args.device,
                          verbose=False)[0]
        d = r.boxes
        xyxy = d.xyxy.cpu().numpy() if d is not None and len(d) else np.zeros((0, 4))
        conf = d.conf.cpu().numpy() if d is not None and len(d) else np.zeros((0,))
        dets = np.hstack([xyxy, conf[:, None]]) if len(conf) else np.zeros((0, 5))
        rows.append({"image": stem, "group": group_key(stem), "is_logo_image": bool(is_logo),
                     "sub": sub, "split": sp, "n_gt": len(gts), "img_area": img_area,
                     "max_conf": float(conf.max()) if len(conf) else 0.0,
                     "dets": dets, "gts": gts})

    is_logo = np.array([r["is_logo_image"] for r in rows])
    maxc = np.array([r["max_conf"] for r in rows])
    gt_fracs = frame_area_fracs(rows)
    if gt_fracs.size:
        q = lambda p: round(float(np.percentile(gt_fracs, p)), 5)
        print("\n=== reviewed ground truth: box area as a FRACTION of frame ===")
        print(f"  boxes {gt_fracs.size}  p10 {q(10)}  p50 {q(50)}  p90 {q(90)}")
        print(f"  frac of boxes under 1% of frame = "
              f"{float((gt_fracs < 0.01).mean()):.4f}   <- the number that decides")
        print("  whether this video set can train a tiny-object detector at all.")
        if float((gt_fracs < 0.01).mean()) < 0.05:
            print("  WARNING: sub-1% boxes are nearly absent from the reviewed set.")
            print("           Training on it will reproduce tiny recall = 0.")
    # A root with no negatives has an empty class; mean/percentile of an
    # empty slice is nan or an IndexError, and the separation numbers are then
    # undefined rather than zero. Say so instead of printing a number.
    n_pos, n_neg = int(is_logo.sum()), int((~is_logo).sum())
    if not n_neg:
        print("\n=== confidence separation (val) ===")
        print(f"  logo images     n={n_pos:>3}  max-conf mean {maxc[is_logo].mean():.4f} "
              f"median {np.median(maxc[is_logo]):.4f}  p90 {np.percentile(maxc[is_logo],90):.4f}")
        print("  NON-logo images n=  0  — no negatives in this root, so AUROC /")
        print("                       FP-per-nonlogo are undefined. Run")
        print("                       scripts/mine_logo_negatives.py first.")
    else:
        print(f"\n=== confidence separation ({args.split}) ===")
        print(f"  logo images     n={n_pos:>3}  max-conf mean {maxc[is_logo].mean():.4f} "
              f"median {np.median(maxc[is_logo]):.4f}  p90 {np.percentile(maxc[is_logo],90):.4f}")
        print(f"  NON-logo images n={n_neg:>3}  max-conf mean {maxc[~is_logo].mean():.4f} "
              f"median {np.median(maxc[~is_logo]):.4f}  p90 {np.percentile(maxc[~is_logo],90):.4f}")
    auc = _auroc(maxc, is_logo) if n_neg else None
    print(f"  AUROC (logo vs non-logo, per-image max conf) = "
          f"{'undefined (no negatives)' if auc is None else auc}")

    # PR-AUC over every box, not per-image max-conf: a logo image whose only
    # mark sits at 0.2 conf is a real miss that max-conf hides whenever the
    # same image also carries a 0.9 box somewhere else.
    scores, labs = [], []
    for r in rows:
        for i in range(len(r["dets"])):
            scores.append(float(r["dets"][i, 4]))
            labs.append(1.0 if _matched(r["dets"][i, :4], r["gts"]) else 0.0)
    prauc = (float(average_precision_score(labs, scores))
             if scores and any(labs) else None)
    print(f"  PR-AUC (box-level, IoU 0.5 matched) = "
          f"{None if prauc is None else round(prauc, 4)}   (baseline = "
          f"{None if prauc is None else round(float(np.mean(labs)), 4)} positive rate)")

    print("\n=== threshold sweep (IoU 0.5) ===")
    print(f"  {'conf':>6} {'P':>7} {'R':>7} {'F1':>7} {'TP/img':>8} {'FP/img':>8} "
          f"{'FN/img':>8} {'FP/nonlogo':>11}")
    sweep = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70):
        tp = fp = ngt = 0
        fp_neg = 0
        for r in rows:
            keep = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
            t, n = iou_match(keep, r["gts"])
            tp += t
            fp += n - t
            ngt += r["n_gt"]
            if not r["is_logo_image"]:
                fp_neg += n
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / ngt if ngt else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        nimg = len(rows)
        sweep.append({"conf": thr, "precision": round(P, 4), "recall": round(R, 4),
                      "f1": round(F, 4), "tp_per_image": round(tp / nimg, 2),
                      "fp_per_image": round(fp / nimg, 2),
                      "fn_per_image": round(max(0, ngt - tp) / nimg, 2),
                      "fp_per_nonlogo_image": round(fp_neg / max(1, int((~is_logo).sum())), 2)})
        print(f"  {thr:>6.2f} {P:>7.4f} {R:>7.4f} {F:>7.4f} {tp/nimg:>8.2f} {fp/nimg:>8.2f} "
              f"{(ngt-tp)/nimg:>8.2f} {fp_neg/max(1,int((~is_logo).sum())):>11.2f}")

    best = max(sweep, key=lambda s: s["f1"])
    if best["f1"] == 0.0:
        # Every threshold scored F1 0, so max() returns whichever comes first and
        # the A/B table below would be reported at an arbitrary operating point.
        # That is precisely the "detector is bad" case, and a number read off an
        # arbitrary threshold is worse than no number.
        print("\n  best F1 is 0.0 at EVERY threshold - the detector matched nothing.")
        print("  Falling back to conf 0.25 for the table below; treat every rate as")
        print("  a floor, not an operating point. Do not tune against this run.")
        best = dict(next(s for s in sweep if s["conf"] == 0.25), f1=0.0)
    else:
        print(f"\n  best F1 {best['f1']:.4f} at conf {best['conf']:.2f}")

    fired = fired_matrix(rows, best["conf"])
    print("\n=== detector-fired x reviewed-verdict  (the A/B test) ===")
    print(f"  at conf {best['conf']:.2f}   logo frames {fired['logo_frames']}  "
          f"logo-free frames {fired['logo_free_frames']}")
    print(f"  {'':<22}{'human: HAS LOGO':>18}{'human: NO LOGO':>18}")
    print(f"  {'detector FIRED':<22}{fired['fired_on_logo']:>18}{fired['fired_on_free']:>18}")
    print(f"  {'detector SILENT':<22}{fired['silent_on_logo']:>18}{fired['silent_on_free']:>18}")
    print(f"\n  logo-frame recall  {fired['logo_frame_recall']:.4f}"
          f"   ({fired['fired_on_logo']}/{fired['logo_frames']} logo frames got >=1 box)")
    print(f"  FP / logo-free frame  {fired['fp_per_logo_free_frame']:.4f}"
          f"   ({fired['fired_on_free']}/{fired['logo_free_frames']} empty frames got a box)")
    print(f"  logo-BOX recall (best-F1 conf)  {fired['box_recall']:.4f}")
    print("  A) 328 empty frames were mostly NO LOGO -> fired_on_free is low")
    print("  B) they were missed logos        -> silent_on_logo is high")

    sizes = size_breakdown(rows, best["conf"])
    print("\n=== per object size (GT area in px, best-F1 conf) ===")
    print(f"  {'bucket':>7} {'GT':>5} {'TP':>5} {'FN':>5} {'recall':>7} {'med px^2':>9} "
          f"{'med area/img':>13}")
    for b, s in sizes.items():
        if not s["gt"]:
            print(f"  {b:>7} {0:>5} {0:>5} {0:>5} {'-':>7} {'-':>9} {'-':>13}")
            continue
        print(f"  {b:>7} {s['gt']:>5} {s['tp']:>5} {s['fn']:>5} {s['recall']:>7.4f} "
              f"{s['median_px2']:>9.0f} {s['median_area_frac']:>13.5f}")
    all_frac = []
    for r in rows:
        if len(r["gts"]):
            g = r["gts"]
            all_frac.extend(((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
                             / r["img_area"]).tolist())
    med_frac = float(np.median(all_frac)) if all_frac else None
    print(f"\n  median GT bbox area / image area = "
          f"{'n/a (no GT)' if med_frac is None else f'{med_frac:.5f}'}")

    groups = group_breakdown(rows, best["conf"], box)
    scored = [g for g in groups if g["gt"] or g["tp"]]
    print("\n=== per source group (video) ===")
    if len(groups) > 12:
        # One group per image means the source has no video structure (a photo
        # dataset), so a 300-row table is noise. Summarise; JSON keeps the rows.
        f1s = [g["f1"] for g in scored]
        print(f"  {len(groups)} groups, {len(scored)} with ground truth — no video")
        print("  structure in this root, so per-video is per-image. Distribution:")
        if f1s:
            print(f"    F1   min {min(f1s):.4f}  p25 {np.percentile(f1s,25):.4f}  "
                  f"median {np.median(f1s):.4f}  p75 {np.percentile(f1s,75):.4f}  max {max(f1s):.4f}")
            print(f"    groups with F1 == 0: {sum(1 for f in f1s if f == 0)}/{len(f1s)}")
        print(f"    worst 5: {[(g['group'], g['f1']) for g in sorted(scored, key=lambda g: g['f1'])[:5]]}")
        print(f"    best  5: {[(g['group'], g['f1']) for g in sorted(scored, key=lambda g: -g['f1'])[:5]]}")
        print("    (full table in the --out json)")
    else:
        print(f"  {'group':>12} {'img':>4} {'GT':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'P':>7} {'R':>7} {'F1':>7} {'mAP50*':>8}")
        for g in groups:
            print(f"  {str(g['group'])[:12]:>12} {g['images']:>4} {g['gt']:>5} {g['tp']:>4} "
                  f"{g['fp']:>4} {g['fn']:>4} {g['precision']:>7.4f} {g['recall']:>7.4f} "
                  f"{g['f1']:>7.4f} {g['map50_star']:>8.4f}")
        f1s = [g["f1"] for g in scored]
        if len(f1s) > 1 and max(f1s) > 0:
            # A group can score a perfect 0.0, which makes the spread ratio
            # infinite. Report it as unbounded instead of dividing by zero and
            # killing the run before the JSON is written.
            lo = min(f1s)
            spread = f"{max(f1s)/lo:.1f}x" if lo > 0 else "unbounded (a group scored 0)"
            print(f"\n  F1 across groups: min {lo:.4f}  max {max(f1s):.4f}  "
                  f"ratio {spread}  <- a pooled F1 hides this spread")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": args.model, "split": args.split, "root": str(root),
            "map50": round(float(box.map50), 4), "map50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "auroc_logo_vs_nonlogo_maxconf": auc,
            "pr_auc_box_level": None if prauc is None else round(prauc, 4),
            "logo_maxconf_mean": round(float(maxc[is_logo].mean()), 4) if is_logo.any() else None,
            "nonlogo_maxconf_mean": round(float(maxc[~is_logo].mean()), 4) if (~is_logo).any() else None,
            "median_gt_bbox_area_frac": None if med_frac is None else round(med_frac, 5),
            "sweep": sweep, "best_f1": best,
            "fired_matrix": fired,
            "gt_area_frac": {
                "boxes": int(gt_fracs.size),
                "p10": round(float(np.percentile(gt_fracs, 10)), 5) if gt_fracs.size else None,
                "p50": round(float(np.percentile(gt_fracs, 50)), 5) if gt_fracs.size else None,
                "p90": round(float(np.percentile(gt_fracs, 90)), 5) if gt_fracs.size else None,
                "frac_under_1pct_frame": round(float((gt_fracs < 0.01).mean()), 4) if gt_fracs.size else None,
            },
            "per_size": sizes, "per_group": groups,
        }, indent=2))
        print(f"  -> {args.out}")
    return 0


def frame_area_fracs(rows: list[dict]) -> np.ndarray:
    """Every reviewed GT box as a fraction of its own frame.

    COCO's tiny/small/medium buckets are absolute pixels, so they mean different
    things on a 640x360 frame and a 720x1280 one. The failure under test is
    stated as a fraction of the frame, so it has to be measured that way.
    """
    out = []
    for r in rows:
        g = r["gts"]
        if len(g):
            out.extend((((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))
                        / max(1.0, r["img_area"])).tolist())
    return np.array(out)


def fired_matrix(rows: list[dict], thr: float) -> dict:
    """2x2 of detector-fired against the human verdict, plus the two rates.

    Pooled recall cannot separate "fires on everything" from "finds logos": a
    detector with recall 1.0 and 70 FP/frame has the same box recall as a perfect
    one. Splitting by whether the frame actually has a logo is what tells them
    apart, and it is the only way to resolve why silent frames are silent.
    """
    fired_on_logo = fired_on_free = silent_on_logo = silent_on_free = 0
    tp = ngt = 0
    for r in rows:
        dets = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
        said = len(dets) > 0
        if r["n_gt"]:
            fired_on_logo += said
            silent_on_logo += not said
            t, _ = iou_match(dets, r["gts"])
            tp += t
            ngt += r["n_gt"]
        else:
            fired_on_free += said
            silent_on_free += not said
    logo_frames = fired_on_logo + silent_on_logo
    free_frames = fired_on_free + silent_on_free
    return {
        "conf": thr,
        "logo_frames": logo_frames, "logo_free_frames": free_frames,
        "fired_on_logo": int(fired_on_logo), "silent_on_logo": int(silent_on_logo),
        "fired_on_free": int(fired_on_free), "silent_on_free": int(silent_on_free),
        "logo_frame_recall": fired_on_logo / logo_frames if logo_frames else None,
        "fp_per_logo_free_frame": fired_on_free / free_frames if free_frames else None,
        "box_recall": tp / ngt if ngt else None,
        "boxes": ngt, "tp": tp,
    }


def _matched(box: np.ndarray, gts: np.ndarray) -> bool:
    if not len(gts):
        return False
    inter = (np.maximum(0.0, np.minimum(box[2], gts[:, 2]) - np.maximum(box[0], gts[:, 0]))
             * np.maximum(0.0, np.minimum(box[3], gts[:, 3]) - np.maximum(box[1], gts[:, 1]))).sum()
    a = (box[2] - box[0]) * (box[3] - box[1])
    g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    return bool(((inter / (a + g - inter)) >= 0.5).any())


def size_breakdown(rows: list[dict], thr: float) -> dict:
    """Recall per GT object size. A tiny logo missed is invisible in mAP50-95
    if the frame also holds one large mark, which is exactly the video case."""
    out = {k: {"gt": 0, "tp": 0, "fn": 0, "recall": 0.0, "px2": [], "frac": []}
           for k in (*SIZE_EDGES_PX, "large")}
    for r in rows:
        g = r["gts"]
        if not len(g):
            continue
        dets = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
        used, hits = set(), {}
        for di in np.argsort(-dets[:, 4]) if len(dets) else []:
            best, bj = 0.5, -1
            for gj, gt in enumerate(g):
                if gj in used:
                    continue
                ix1, iy1 = max(dets[di, 0], gt[0]), max(dets[di, 1], gt[1])
                ix2, iy2 = min(dets[di, 2], gt[2]), min(dets[di, 3], gt[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                u = ((dets[di, 2] - dets[di, 0]) * (dets[di, 3] - dets[di, 1])
                     + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter)
                v = inter / u if u > 0 else 0.0
                if v >= best:
                    best, bj = v, gj
            if bj >= 0:
                used.add(bj)
                hits[bj] = True
        img_area = r["img_area"]
        for gj, gt in enumerate(g):
            bw, bh = gt[2] - gt[0], gt[3] - gt[1]
            b = size_bucket(bw * bh)
            out[b]["gt"] += 1
            out[b]["tp"] += int(hits.get(gj, False))
            out[b]["px2"].append(bw * bh)
            out[b]["frac"].append(bw * bh / max(1.0, img_area))
    for b, s in out.items():
        s["fn"] = s["gt"] - s["tp"]
        s["recall"] = s["tp"] / s["gt"] if s["gt"] else 0.0
        s["median_px2"] = float(np.median(s["px2"])) if s["px2"] else 0.0
        s["median_area_frac"] = float(np.median(s["frac"])) if s["frac"] else 0.0
        s.pop("px2"), s.pop("frac")
    return out


def group_breakdown(rows: list[dict], thr: float, box) -> list[dict]:
    """Per source video. `map50_star` is box-map at the single operating
    threshold, NOT mAP50: mAP50 is defined over a curve and one group has too
    few images to integrate it meaningfully."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    out = []
    for g, rs in sorted(by.items()):
        tp = fp = ngt = 0
        scores, labs = [], []
        for r in rs:
            dets = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
            t, n = iou_match(dets, r["gts"])
            tp += t
            fp += n - t
            ngt += r["n_gt"]
        for r in rs:
            dets = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
            for i in range(len(dets)):
                scores.append(float(dets[i, 4]))
                labs.append(1.0 if _matched(dets[i, :4], r["gts"]) else 0.0)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / ngt if ngt else 0.0
        out.append({
            "group": g, "images": len(rs), "gt": ngt, "tp": tp, "fp": fp, "fn": ngt - tp,
            "precision": round(P, 4), "recall": round(R, 4),
            "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
            "map50_star": round(float(average_precision_score(labs, scores)), 4)
            if scores and any(labs) else 0.0,
        })
    return out


if __name__ == "__main__":
    sys.exit(main())
