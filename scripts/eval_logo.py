"""
Measure logo DETECTION and brand IDENTIFICATION on the real eval set.

This is the measurement that never existed. It runs the real detector + the
real BrandResolver over real LogoDet-3K frames and reports, at several
confidence thresholds:

  precision / recall / F1 / mAP50   -- did we FIND the logo boxes?
  brand_accuracy                    -- of the boxes we matched, how many did we
                                     name correctly? (the 0/13 number)
  false_brand_rate                  -- of accepted names, how many are wrong?
  abstention                        -- negatives where we correctly said nothing

Predictions are written to <out>/predictions.json so every number is
reproducible and auditable after the fact, instead of living in a log line.

Usage
-----
    python scripts/eval_logo.py --eval-set benchmark/eval_real --split test
    python scripts/eval_logo.py --eval-set benchmark/eval_real --split test \
        --ablate-brand-queries          # compare the old 45-class behaviour
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _ap(scores, n_gt, n_gt_idx) -> float:
    """Average precision via the standard interpolated precision-recall curve."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    hits = np.asarray(n_gt_idx)[order] > 0
    if not hits.any():
        return 0.0
    ctp = np.cumsum(hits)
    cfp = np.cumsum(~hits)
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / n_gt
    # Monotone-decreasing precision envelope, then integrate over recall steps.
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _auroc(scores, labels) -> Optional[float]:
    """AUROC via the rank-sum identity, with ties averaged. None if one class is empty.

    Ties matter: argsort-of-argsort assigns distinct ranks to equal scores in
    index order, so an all-tied input scores 1.0 instead of 0.5 and any score
    with many ties is biased by row order.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return round(float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def _identity_funnel(pred_rows) -> Dict[str, int]:
    """Why each detection did or didn't become a brand.

    Brand accuracy on its own is uninterpretable when the resolver abstains on
    almost everything: 3-of-3 looks perfect and 1-of-3 looks terrible, and
    neither says the detector worked. The funnel does.
    """
    funnel: Dict[str, int] = defaultdict(int)
    for rec in pred_rows:
        for d in rec["dets"]:
            if d.get("brand"):
                funnel[f"named:{d.get('resolution_source') or 'unknown'}"] += 1
            else:
                funnel[f"unresolved:{d.get('unresolved_reason') or 'untagged'}"] += 1
    return dict(sorted(funnel.items()))


def evaluate(pred_rows, gold, iou_thr=0.5, thresholds=(0.10, 0.20, 0.30, 0.40, 0.50)):
    """pred_rows: [{image, dets:[{bbox,conf,brand}]}]; gold: {image: [boxes]}."""
    report = {}
    for thr in thresholds:
        tp = fp = fn = 0
        brand_ok = brand_named = 0
        brand_wrong = []
        for rec in pred_rows:
            gts = gold.get(rec["image"], [])
            dets = [d for d in rec["dets"] if d["conf"] >= thr]
            # Confidence order inside the image, so the strongest box claims the
            # GT. Input order lets a low-confidence box steal it.
            used = [False] * len(gts)
            for d in sorted(dets, key=lambda x: -x["conf"]):
                best, bi = 0.0, -1
                for i, g in enumerate(gts):
                    if used[i]:
                        continue
                    v = _iou(d["bbox"], g["bbox"])
                    if v > best:
                        best, bi = v, i
                if best >= iou_thr and bi >= 0:
                    used[bi] = True
                    tp += 1
                    named = d.get("brand")
                    if named:
                        brand_named += 1
                        if named == gts[bi]["brand"]:
                            brand_ok += 1
                        else:
                            brand_wrong.append((rec["image"], named, gts[bi]["brand"]))
                else:
                    fp += 1
            fn += sum(1 for u in used if not u)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        report[f"conf>={thr:.2f}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "boxes_named": brand_named,
            "brand_accuracy": round(brand_ok / brand_named, 4) if brand_named else None,
            "false_brand_rate": round(1 - brand_ok / brand_named, 4) if brand_named else None,
        }
    return report, brand_wrong


def mean_ap(pred_rows, gold, iou_thr=0.5):
    """mAP50 over one class, ranked across ALL confidences (not per-threshold).

    The per-threshold rows above are point metrics; averaging them would be
    meaningless. This walks the global confidence ranking once, so a detector
    that is right only at low confidence is not credited for it.
    """
    scores, hits, n_gt = [], [], 0
    for rec in pred_rows:
        gts = gold.get(rec["image"], [])
        n_gt += len(gts)
        used = [False] * len(gts)
        # High confidence first inside the image so the best box claims the GT.
        for d in sorted(rec["dets"], key=lambda x: -x["conf"]):
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = _iou(d["bbox"], g["bbox"])
                if v > best:
                    best, bi = v, i
            if best >= iou_thr and bi >= 0:
                used[bi] = True
                hits.append(1)
            else:
                hits.append(0)
            scores.append(d["conf"])
    ap = _ap(scores, n_gt, hits)
    return round(ap, 4) if ap == ap else None, n_gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="benchmark/eval_real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="defaults to <eval-set>/predictions_<split>.json")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--no-resolve", action="store_true",
                    help="detector only; skip BrandResolver identity")
    ap.add_argument("--ablate-brand-queries", action="store_true",
                    help="re-add the 39 '<Brand> logo' prompts (the old behaviour)")
    ap.add_argument("--backend", choices=["yolo", "yolo_world"], default=None,
                    help="override config backend so A/B runs share one eval set")
    ap.add_argument("--model", default=None, help="override model weights")
    ap.add_argument("--tag", default=None, help="label for the report header")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop detections below this before identity resolution. "
                         "OCR runs per crop, so a 0.10 sweep on dense frames is "
                         "minutes of PaddleOCR for boxes the operating point "
                         "would never use.")
    args = ap.parse_args()

    import cv2

    root = Path(args.eval_set)
    labels = json.load(open(root / "labels.json"))
    rows = labels[args.split]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit(f"FATAL: split {args.split!r} is empty in {root/'labels.json'}")

    from src.pipeline import Phase1Pipeline

    pipeline = Phase1Pipeline()
    if args.backend:
        pipeline.cfg["layer1"]["logo_detection"]["backend"] = args.backend
    if args.model:
        pipeline.cfg["layer1"]["logo_detection"]["model"] = args.model
    if args.ablate_brand_queries:
        from src.brand_catalog import build_text_queries

        logo_cfg = pipeline.cfg["layer1"]["logo_detection"]
        logo_cfg["region_proposal_only"] = False
        mode = "brand prompts ON (legacy)"
    else:
        logo_cfg = pipeline.cfg["layer1"]["logo_detection"]
        mode = ("region-proposal-only" if logo_cfg.get("region_proposal_only", True)
                else "brand prompts ON (from config)")

    # Touch the lazy property, then report what the detector ACTUALLY got —
    # never infer the class count from config, which is what made an earlier
    # run print "6 classes" while 45 were loaded.
    probe = pipeline.logo_detector
    n_classes = len(getattr(probe, "_current_queries", []) or [])
    if n_classes:
        print(f"{mode}: {n_classes} classes")
    else:
        print(f"{mode}: trained detector, no text prompts "
              f"(model={pipeline.cfg['layer1']['logo_detection']['model']})")
    if args.tag:
        print(f"--- {args.tag} ---")

    resolver = None
    if not args.no_resolve:
        from src.layer2.brand_resolver import BrandResolver

        _lr_cfg = pipeline.cfg["layer1"].get("logo_retrieval", {})
        # MUST pass the pipeline's retrieval index. Without it the resolver has
        # no CLIP bank and lazily fetches model weights on the first crop, which
        # stalls the whole run at 0% CPU. Same wiring as pipeline.py:1623.
        resolver = BrandResolver(
            ocr_extractor=getattr(pipeline, "ocr", None),
            class_confidence=pipeline.cfg["layer1"]["logo_detection"]["class_confidence"],
            crop_scale=pipeline.cfg["layer1"]["logo_detection"]["crop_scale"],
            retrieval_index=getattr(pipeline, "logo_retrieval", None) if _lr_cfg.get("enabled", True) else None,
            retrieval_min_similarity=_lr_cfg.get("min_similarity", 0.22),
            retrieval_min_margin=_lr_cfg.get("min_margin", 0.10),
            max_logo_area_fraction=pipeline.cfg["layer1"]["logo_detection"]["max_logo_area_fraction"],
        )

    pred_rows = []
    gold = {}
    t0 = time.monotonic()
    for i, r in enumerate(rows):
        img_bgr = cv2.imread(str(root / "images" / args.split / r["image"]))
        if img_bgr is None:
            continue
        # detect() and BrandResolver both take RGB frames (the pipeline feeds
        # them RGB and each converts internally). imread returns BGR, so every
        # score from this script was computed on colour-swapped input.
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gold[r["image"]] = r["boxes"]
        dets = pipeline.logo_detector.detect(img) or []
        if args.min_conf:
            dets = [d for d in dets if d.get("confidence", 0.0) >= args.min_conf]
        out = []
        for d in dets:
            rec = {"bbox": [float(v) for v in d["bbox"]], "conf": float(d["confidence"]),
                   "prompt": d.get("text_prompt")}
            if resolver is not None:
                # resolve() takes per-frame lists and returns per-frame lists.
                res = resolver.resolve([[d]], [img])[0][0]
                rec["brand"] = res.get("brand")
                rec["class_unconfirmed"] = bool(res.get("class_unconfirmed"))
                # Why it did or didn't name a brand. Without these, an unresolved
                # crop is indistinguishable from a detector false positive and
                # identity accuracy is reported on the handful that resolved.
                rec["unresolved_reason"] = res.get("unresolved_reason")
                rec["resolution_source"] = res.get("resolution_source")
                rec["retrieval_similarity"] = res.get("retrieval_similarity")
                rec["retrieval_top3"] = res.get("retrieval_top3")
                rec["retrieval_diag"] = res.get("retrieval_diag")
                rec["ocr_text"] = res.get("ocr_text")
            out.append(rec)
        pred_rows.append({"image": r["image"], "negative": r["negative"], "dets": out})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  {time.monotonic()-t0:.1f}s")

    elapsed = time.monotonic() - t0
    report, wrong = evaluate(pred_rows, gold, iou_thr=args.iou)
    mAP50, n_gt = mean_ap(pred_rows, gold, iou_thr=args.iou)

    # Abstention: negatives where nothing was named.
    neg = [r for r in pred_rows if r["negative"]]
    neg_named = sum(1 for r in neg for d in r["dets"] if d.get("brand"))
    summary = {
        "split": args.split,
        "images": len(pred_rows),
        "positive_images": sum(1 for r in pred_rows if not r["negative"]),
        "negative_images": len(neg),
        "gt_boxes": sum(len(v) for v in gold.values()),
        "detections_raw": sum(len(r["dets"]) for r in pred_rows),
        "elapsed_s": round(elapsed, 1),
        "ms_per_image": round(1000 * elapsed / max(1, len(pred_rows)), 1),
        "identity_resolver": not args.no_resolve,
        "brand_queries_included": bool(args.ablate_brand_queries),
        "negative_images_with_a_named_brand": neg_named,
        "identity_funnel": _identity_funnel(pred_rows),
        "open_set": {
            "note": "negatives in this eval set are REAL logos whose brand is "
                    "outside the catalog, so detector confidence separating "
                    "them from known-brand images is the open-set score.",
            "auroc_per_detection": _auroc(
                [d["conf"] for r in pred_rows for d in r["dets"]],
                [r["negative"] for r in pred_rows for _ in r["dets"]],
            ),
            "auroc_image_max_conf": _auroc(
                [max((d["conf"] for d in r["dets"]), default=0.0) for r in pred_rows],
                [r["negative"] for r in pred_rows],
            ),
            "named_on_unknown": sum(
                1 for r in pred_rows if r["negative"] for d in r["dets"] if d.get("brand")
            ),
            "unknown_detections": sum(len(r["dets"]) for r in pred_rows if r["negative"]),
        },
        "mAP50": mAP50,
        "min_conf_for_identity": args.min_conf,
        "iou_threshold": args.iou,
        "thresholds": report,
        "brand_mismatches_sample": wrong[:25],
    }
    out = Path(args.out or root / f"predictions_{args.split}.json")
    json.dump({"summary": summary, "predictions": pred_rows}, open(out, "w"), indent=1)

    print(f"\n{'thr':>10} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'named':>6} {'brandacc':>9}")
    for k, v in report.items():
        ba = "-" if v["brand_accuracy"] is None else f"{v['brand_accuracy']:.3f}"
        print(f"{k:>10} {v['tp']:>4} {v['fp']:>5} {v['fn']:>4} {v['precision']:>7.4f} "
              f"{v['recall']:>7.4f} {v['f1']:>7.4f} {v['boxes_named']:>6} {ba:>9}")
    print(f"\nmAP50 (IoU {args.iou}, global ranking, {n_gt} GT): "
          f"{'n/a' if mAP50 is None else f'{mAP50:.4f}'}")
    print("\nidentity funnel:")
    for k, v in summary["identity_funnel"].items():
        print(f"  {k:44s} {v}")
    os_ = summary["open_set"]
    print(f"\nopen-set: AUROC/detection={os_['auroc_per_detection']}  "
          f"AUROC/image={os_['auroc_image_max_conf']}  "
          f"named on unknown={os_['named_on_unknown']}/{os_['unknown_detections']}")
    print(f"negatives: {len(neg)}  of which produced a named brand: {neg_named}")
    print(f"{elapsed:.1f}s  ({summary['ms_per_image']} ms/image)")
    print(f"predictions -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
