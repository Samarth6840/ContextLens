"""
Phase 1.4b: decisive hero-logo re-analysis of saved OWLv2 predictions.

Reuses scripts/owlv2_recall_test.py predictions WITHOUT re-running inference,
but scores them against a filtered ground truth matching what a creator video
frame looks like: one or a few prominent logos, not a dense shelf photo.

Analyses:
  1. FULL (reference): unfiltered GT, same table as the 1.4 run.
  2. TOP-K HIT RATE: per image keep the largest K GT boxes by area; report
     fraction of images where >=1 det matches a kept box (the production
     question: "if there's an obvious logo in frame, does OWLv2 find it?"),
     plus the standard P/R/F1 table on the filtered GT.
  3. AREA-FILTERED: keep GT boxes with area > --min-area-frac of the frame
     (default 2%), recompute P/R/F1.

Usage:
    python scripts/owlv2_recall_1b4.py --preds benchmark/eval_real/owlv2_preds_test.json
"""
import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = ilu.spec_from_file_location(
    "eval_logo", Path(__file__).resolve().parent / "eval_logo.py"
)
eval_logo = ilu.module_from_spec(_spec)
_spec.loader.exec_module(eval_logo)


def _box_area(b):
    x1, y1, x2, y2 = b["bbox"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _img_size(eval_set, split, image):
    p = Path(eval_set) / "images" / split / image
    with Image.open(p) as im:
        return im.size  # (w, h)


def print_report(title, report, n_gt=None):
    print(f"\n=== {title} ===")
    if n_gt is not None:
        print(f"GT boxes in scope: {n_gt}")
    print(f"{'thr':>10} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for k, v in report.items():
        print(
            f"{k:>10} {v['tp']:>4} {v['fp']:>5} {v['fn']:>4} "
            f"{v['precision']:>7.4f} {v['recall']:>7.4f} {v['f1']:>7.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="benchmark/eval_real/owlv2_preds_test.json")
    ap.add_argument("--eval-set", default="benchmark/eval_real")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    args = ap.parse_args()

    saved = json.load(open(args.preds))
    pred_rows = saved["pred_rows"]
    gold = saved["gold"]
    split = saved.get("split", "test")
    thresholds = (0.10, 0.20, 0.30, 0.40, 0.50)

    # 1. Full reference.
    report, _ = eval_logo.evaluate(pred_rows, gold, iou_thr=args.iou)
    n_gt = sum(len(v) for v in gold.values())
    print_report(f"FULL GT (reference, n={n_gt} boxes)", report)

    # Image dims for area fractions.
    dims = {img: _img_size(args.eval_set, split, img) for img in gold}

    # 2. Top-K largest boxes per image.
    top_gold = {}
    for img, boxes in gold.items():
        ranked = sorted(boxes, key=_box_area, reverse=True)
        top_gold[img] = ranked[: args.top_k]
    n_top = sum(len(v) for v in top_gold.values())
    report, _ = eval_logo.evaluate(pred_rows, top_gold, iou_thr=args.iou)
    print_report(f"TOP-{args.top_k} LARGEST GT PER IMAGE (n={n_top} boxes)", report)

    # Hit rate: fraction of images where >=1 det (conf>=thr) matches a kept box.
    print(f"\n--- top-{args.top_k} HIT RATE (image with >=1 matched hero box) ---")
    print(f"{'thr':>10} {'hit':>4} {'/':>2} {'n':>3} {'rate':>7}")
    for thr in thresholds:
        hit = 0
        n = 0
        for rec in pred_rows:
            gts = top_gold.get(rec["image"], [])
            if not gts:
                continue
            n += 1
            dets = sorted(
                [d for d in rec["dets"] if d["conf"] >= thr],
                key=lambda x: -x["conf"],
            )
            matched = any(
                eval_logo._iou(d["bbox"], g["bbox"]) >= args.iou for d in dets for g in gts
            )
            hit += matched
        print(f"conf>={thr:>4.2f} {hit:>4} / {n:>3} {hit/max(1,n):>7.4f}")

    # 3. Area-filtered GT.
    area_gold = {}
    for img, boxes in gold.items():
        w, h = dims[img]
        area = w * h
        area_gold[img] = [b for b in boxes if _box_area(b) / area > args.min_area_frac]
    n_area = sum(len(v) for v in area_gold.values())
    report, _ = eval_logo.evaluate(pred_rows, area_gold, iou_thr=args.iou)
    print_report(
        f"AREA-FILTERED GT (>{args.min_area_frac*100:.0f}% of frame, n={n_area} boxes)",
        report,
    )
    mAP50, n = eval_logo.mean_ap(pred_rows, area_gold, iou_thr=args.iou)
    print(f"\nmAP50 on area-filtered GT ({n} boxes): {mAP50:.4f}")

    # Distribution context: what did the filter actually keep?
    all_fracs = []
    for img, boxes in gold.items():
        w, h = dims[img]
        all_fracs += [_box_area(b) / (w * h) for b in boxes]
    import statistics

    all_fracs.sort()
    print(
        f"\nGT area-frac distribution (n={len(all_fracs)}): "
        f"min {all_fracs[0]:.4f}  p50 {statistics.median(all_fracs):.4f}  "
        f"max {all_fracs[-1]:.4f}  "
        f"frac>{args.min_area_frac}: "
        f"{sum(1 for f in all_fracs if f > args.min_area_frac)/len(all_fracs):.3f}"
    )


if __name__ == "__main__":
    main()
