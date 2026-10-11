"""Phase 1.4b: re-score persisted OWLv2 predictions on a video-like GT subset.

Phase 1.4 ran on dense LogoDet-3K shelf photos (median 6 boxes/image, max 42),
not the sparse hero-logo frames the decision rule assumed. This script loads
the persisted predictions from scripts/owlv2_recall_test.py --out and
re-runs the SAME scoring (scripts/eval_logo.evaluate / mean_ap) against
TWO filtered ground truths, no re-inference needed:

  1. top-k: largest 1 (and separately 2) GT box(es) per image by area --
     "if there's one obvious logo in frame, does OWLv2 find it".
  2. area-filtered: keep only GT boxes above a fraction of frame area
     (--min-area-frac) across the whole set.

Usage:
    python scripts/owlv2_filter_test.py --preds benchmark/eval_real/owlv2_preds_test.json
"""
import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = ilu.spec_from_file_location(
    "eval_logo", Path(__file__).resolve().parent / "eval_logo.py"
)
eval_logo = ilu.module_from_spec(_spec)
_spec.loader.exec_module(eval_logo)


def _area(b):
    bb = b["bbox"] if isinstance(b, dict) else b
    return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])


def _print_report(title, pred_rows, gold, iou):
    report, _ = eval_logo.evaluate(pred_rows, gold, iou_thr=iou)
    mAP50, n_gt = eval_logo.mean_ap(pred_rows, gold, iou_thr=iou)
    print(f"\n== {title} ({n_gt} GT boxes) ==")
    print(f"{'thr':>10} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for k, v in report.items():
        print(
            f"{k:>10} {v['tp']:>4} {v['fp']:>5} {v['fn']:>4} "
            f"{v['precision']:>7.4f} {v['recall']:>7.4f} {v['f1']:>7.4f}"
        )
    print(f"mAP50: {mAP50}")
    return report, mAP50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="persisted owlv2_preds_*.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area-frac", type=float, default=0.02,
                    help="area-filtered variant: keep GT boxes covering at "
                         "least this fraction of frame area")
    args = ap.parse_args()

    blob = json.load(open(args.preds))
    pred_rows = blob["pred_rows"]
    gold_full = blob["gold"]

    # --- variant 1: top-k largest GT boxes per image ("the hero logo") ---
    for k in (1, 2):
        gold = {}
        for img, boxes in gold_full.items():
            ranked = sorted(boxes, key=_area, reverse=True)
            gold[img] = ranked[:k]
        _print_report(f"top-{k} largest GT box(es) per image", pred_rows, gold, args.iou)

    # --- variant 2: absolute area filter (>frac of frame area) ---
    from PIL import Image
    import json as _json

    print("\n(resolving frame sizes for the area filter...)")
    sizes = {}
    # recover image dir from a sibling labels.json if present, else scan
    labels_paths = list(Path(args.preds).parent.glob("labels.json"))
    img_roots = []
    if labels_paths:
        lab = _json.load(open(labels_paths[0]))
        for split_rows in lab.values():
            if isinstance(split_rows, list):
                for r in split_rows:
                    if isinstance(r, dict) and "image" in r:
                        img_roots.append(r["image"])
    # fall back: match prediction image names under */images/*
    cand_dirs = list(Path(args.preds).parent.glob("images/*"))
    found = {}
    for img in gold_full:
        for d in cand_dirs:
            p = d / img
            if p.exists():
                found[img] = p
                break
    for img, p in found.items():
        with Image.open(p) as im:
            sizes[img] = im.size  # (w, h)

    missing = set(gold_full) - set(sizes)
    if missing:
        print(f"  WARNING: no image file for {len(missing)} images; kept unfiltered")
        for img in missing:
            sizes[img] = None

    gold = {}
    kept_total = full_total = 0
    for img, boxes in gold_full.items():
        full_total += len(boxes)
        wh = sizes.get(img)
        if wh is None:
            gold[img] = boxes
            kept_total += len(boxes)
            continue
        frame_area = wh[0] * wh[1]
        kept = [b for b in boxes
                if _area(b["bbox"]) / frame_area >= args.min_area_frac]
        gold[img] = kept
        kept_total += len(kept)
    print(f"area filter >= {args.min_area_frac:.0%}: kept {kept_total}/{full_total} GT boxes")
    _print_report(f"GT boxes >= {args.min_area_frac:.0%} of frame area",
                  pred_rows, gold, args.iou)


if __name__ == "__main__":
    main()
