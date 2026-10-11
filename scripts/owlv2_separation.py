"""Phase 1.5 (evaluation-only): OWLv2 through the separation ledger.

Reuses read_split/group_key/iou_match/fired_matrix/size_breakdown/
group_breakdown/frame_area_fracs/_auroc/_matched from
eval_detector_separation.py verbatim (loaded as a module, no copied
scoring logic) and mean_ap from eval_logo.py. Runs OWLv2 zero-shot
grounding (same 6 generic queries as owlv2_recall_test.py) over the SAME
--root/--split images, prints the SAME sections as the YOLO harness,
plus ms/frame timing.

Differs from the YOLO harness in one honest way: no model.val() mAP --
that comes from Ultralytics internals. mAP50 is via eval_logo.mean_ap.

Usage:
    python scripts/owlv2_separation.py --root weights/logo_detector --split val
    python scripts/owlv2_separation.py --root weights/logo_detector --split val --out benchmark/results/owlv2_separation_val.json
"""
import argparse
import importlib.util as ilu
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SEP = ilu.spec_from_file_location(
    "eval_detector_separation",
    Path(__file__).resolve().parent / "eval_detector_separation.py",
)
sep = ilu.module_from_spec(_SEP)
_SEP.loader.exec_module(sep)

_EV = ilu.spec_from_file_location(
    "eval_logo_mod", Path(__file__).resolve().parent / "eval_logo.py"
)
ev = ilu.module_from_spec(_EV)
_EV.loader.exec_module(ev)

GENERIC_QUERIES = [
    "a brand logo",
    "a company logo",
    "a product logo",
    "a wordmark",
    "a label",
    "a text logo",
]

SWEEP_THRS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def _rows_to_jsonable(rows):
    return [{**r, "dets": np.asarray(r["dets"]).tolist(),
             "gts": np.asarray(r["gts"]).tolist()} for r in rows]


def _rows_from_jsonable(obj):
    return [{**r, "dets": np.asarray(r["dets"], dtype=float),
             "gts": np.asarray(r["gts"], dtype=float)} for r in obj["rows"]]


def run_owlv2(root: Path, items, model_id: str):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {model_id} on {device} ...", flush=True)
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    rows, ms_list = [], []
    for i, (stem, gts, is_logo, img_area, sub, sp) in enumerate(items):
        img_path = root / sub / sp / f"{stem}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            text=[GENERIC_QUERIES], images=image, return_tensors="pt"
        ).to(device)
        ti = time.monotonic()
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.01,
        )[0]
        ms_list.append((time.monotonic() - ti) * 1000.0)
        boxes = results["boxes"].detach().cpu().numpy() \
            if len(results["boxes"]) else np.zeros((0, 4))
        conf = results["scores"].detach().cpu().numpy() \
            if len(results["scores"]) else np.zeros((0,))
        dets = np.hstack([boxes, conf[:, None]]) if len(conf) \
            else np.zeros((0, 5))
        rows.append({
            "image": stem, "group": sep.group_key(stem),
            "is_logo_image": bool(is_logo), "sub": sub, "split": sp,
            "n_gt": len(gts), "img_area": img_area,
            "max_conf": float(conf.max()) if len(conf) else 0.0,
            "dets": dets, "gts": gts,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
    return rows, np.array(ms_list)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="weights/logo_detector")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N images (smoke test).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dets", default=None,
                    help="load raw detections from a *_dets_*.json instead of "
                         "running OWLv2, so this ledger can score ANOTHER "
                         "model's saved predictions (parity check).")
    ap.add_argument("--save-dets", default=None,
                    help="persist raw per-image detections to this path.")
    args = ap.parse_args()

    root = Path(args.root)
    items = sep.read_split(root, args.split)
    print(f"images: {len(items)}")
    if args.limit:
        items = items[: args.limit]
        print(f"  (smoke test: first {len(items)})")

    if args.dets:
        rows = _rows_from_jsonable(json.load(open(args.dets)))
        ms = np.array([])
        print(f"loaded {len(rows)} image rows from {args.dets} (no inference)")
    else:
        rows, ms = run_owlv2(root, items, args.model)
        if args.save_dets:
            Path(args.save_dets).write_text(json.dumps(
                {"model": args.model, "split": args.split, "root": str(root),
                 "rows": _rows_to_jsonable(rows)}))
            print(f"  dets -> {args.save_dets}")
    if ms.size:
        print(f"\nms/frame: mean {ms.mean():.0f}  median {np.median(ms):.0f}  "
              f"p90 {np.percentile(ms, 90):.0f}  (n={len(ms)})")

    is_logo = np.array([r["is_logo_image"] for r in rows])
    maxc = np.array([r["max_conf"] for r in rows])
    n_pos, n_neg = int(is_logo.sum()), int((~is_logo).sum())
    print(f"\n=== confidence separation ({args.split}) ===")
    print(f"  logo images     n={n_pos:>3}")
    print(f"  NON-logo images n={n_neg:>3}")
    auc = sep._auroc(maxc, is_logo) if (n_neg and n_pos) else None
    print(f"  AUROC (logo vs non-logo, per-image max conf) = "
          f"{'undefined (no negatives)' if auc is None else round(auc, 4)}")

    gt_fracs = sep.frame_area_fracs(rows)
    if gt_fracs.size:
        q = lambda p: round(float(np.percentile(gt_fracs, p)), 5)
        print(f"\n=== reviewed ground truth: box area as FRACTION of frame ===")
        print(f"  boxes {gt_fracs.size}  p10 {q(10)}  p50 {q(50)}  p90 {q(90)}")
        print(f"  frac of boxes under 1% of frame = "
              f"{float((gt_fracs < 0.01).mean()):.4f}")

    print("\n=== threshold sweep (IoU 0.5) ===")
    print(f"  {'conf':>6} {'P':>7} {'R':>7} {'F1':>7} {'TP/img':>8} {'FP/img':>8} "
          f"{'FN/img':>8} {'FP/nonlogo':>11}")
    sweep = []
    for thr in SWEEP_THRS:
        tp = fp = ngt = 0
        fp_neg = 0
        for r in rows:
            keep = r["dets"][r["dets"][:, 4] >= thr] if len(r["dets"]) else r["dets"]
            t, n = sep.iou_match(keep, r["gts"])
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
                      "fp_per_nonlogo_image": round(fp_neg / max(1, n_neg), 2)})
        print(f"  {thr:>6.2f} {P:>7.4f} {R:>7.4f} {F:>7.4f} {tp/nimg:>8.2f} {fp/nimg:>8.2f} "
              f"{(ngt-tp)/nimg:>8.2f} {fp_neg/max(1,n_neg):>11.2f}")

    best = max(sweep, key=lambda s: s["f1"])
    if best["f1"] == 0.0:
        print("\n  best F1 is 0.0 at EVERY threshold - matched nothing.")
        best = dict(next(s for s in sweep if s["conf"] == 0.25), f1=0.0)
    else:
        print(f"\n  best F1 {best['f1']:.4f} at conf {best['conf']:.2f}")

    fired = sep.fired_matrix(rows, best["conf"])
    print("\n=== detector-fired x reviewed-verdict (the A/B test) ===")
    print(f"  at conf {best['conf']:.2f}   logo frames {fired['logo_frames']}  "
          f"logo-free frames {fired['logo_free_frames']}")
    print(f"  {'':<22}{'human: HAS LOGO':>18}{'human: NO LOGO':>18}")
    print(f"  {'detector FIRED':<22}{fired['fired_on_logo']:>18}{fired['fired_on_free']:>18}")
    print(f"  {'detector SILENT':<22}{fired['silent_on_logo']:>18}{fired['silent_on_free']:>18}")
    print(f"\n  logo-frame recall  "
          f"{fired['logo_frame_recall'] if fired['logo_frame_recall'] is None else round(fired['logo_frame_recall'], 4)}")
    fplf = fired['fp_per_logo_free_frame']
    print(f"  FP / logo-free frame  "
          f"{'undefined (no logo-free frames)' if fplf is None else round(fplf, 4)}")
    print(f"  logo-BOX recall (best-F1 conf)  "
          f"{fired['box_recall'] if fired['box_recall'] is None else round(fired['box_recall'], 4)}")

    sizes = sep.size_breakdown(rows, best["conf"])
    print("\n=== per object size (GT area in px, best-F1 conf) ===")
    for b, s in sizes.items():
        if not s["gt"]:
            print(f"  {b:>7} {'(no GT)'}")
            continue
        print(f"  {b:>7} {s['gt']:>5} {s['tp']:>5} {s['fn']:>5} {s['recall']:>7.4f} "
              f"{s['median_px2']:>9.0f} {s['median_area_frac']:>13.5f}")

    pred_rows = [{"image": r["image"], "negative": not r["is_logo_image"],
                   "dets": [{"bbox": d[:4].tolist(), "conf": float(d[4])}
                             for d in r["dets"]]} for r in rows]
    gold = {r["image"]: [{"bbox": g.tolist()} for g in r["gts"]] for r in rows}
    mAP50, n_gt = ev.mean_ap(pred_rows, gold, iou_thr=0.5)
    print(f"\nmAP50 ({n_gt} GT boxes, via eval_logo.mean_ap): {round(mAP50, 4)}")

    groups = sep.group_breakdown(rows, best["conf"], None)
    scored = [g for g in groups if g["gt"] or g["tp"]]
    print("\n=== per source group ===")
    if len(groups) > 12:
        f1s = [g["f1"] for g in scored]
        print(f"  {len(groups)} groups, {len(scored)} with GT -- per-image here.")
        if f1s:
            print(f"    F1 min {min(f1s):.4f} median {np.median(f1s):.4f} "
                  f"max {max(f1s):.4f}; F1==0: "
                  f"{sum(1 for f in f1s if f == 0)}/{len(f1s)}")
    else:
        for g in groups:
            print(f"  {str(g['group'])[:12]:>12} img={g['images']} gt={g['gt']} "
                  f"tp={g['tp']} fp={g['fp']} fn={g['fn']} F1={g['f1']:.4f}")

    if args.out:
        ms_sum = {"mean": 0.0, "median": 0.0, "p90": 0.0}
        if ms.size:
            ms_sum = {"mean": round(float(ms.mean()), 1),
                      "median": round(float(np.median(ms)), 1),
                      "p90": round(float(np.percentile(ms, 90)), 1)}
        Path(args.out).write_text(json.dumps({
            "model": args.model, "split": args.split, "root": str(root),
            "note": "OWLv2 separation: no Ultralytics val(); mAP50 via eval_logo.mean_ap.",
            "ms_per_frame": ms_sum,
            "map50": mAP50, "auroc": auc,
            "sweep": sweep, "best_f1": best,
            "fired_matrix": fired,
            "per_size": sizes, "per_group": groups,
        }, indent=2, default=float))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
