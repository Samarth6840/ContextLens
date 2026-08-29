"""
Logo detection benchmark metrics.

Standard object-detection evaluation on logo detections:

  * IoU matching at a fixed threshold (default 0.5).
  * Per-brand and aggregate Precision / Recall / F1.
  * mAP@0.5 computed over a confidence-threshold sweep (per-brand PR curves,
    mean over brands — COCO-style 'mAP' without the 0.5:0.95 IoU ladder).

Detection format (both GT and predictions):
    {"bbox": [x1, y1, x2, y2], "brand": "NIKE"}   # GT
    {"bbox": [...], "confidence": 0.61}            # predictions also carry conf

Brand matching is IoU-based localization (a prediction counts as a true
positive when it overlaps a GT box by >= iou_threshold). Brand ATTRIBUTION is
reported separately as `brand_accuracy` — the fraction of true-positive boxes
whose resolved brand equals the GT brand. Keeping localization and attribution
separate is the honest split for a logo-retrieval pipeline: a detector may find
logo regions without knowing which brand they are, and attribution failures are
visible instead of silently inflating or deflating P/R/F1.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_predictions(
    gt_boxes: List[List[float]], gt_brands: List[str], preds: List[dict], iou_thresh: float
):
    """Greedy IoU matching; returns (tp_flags, matched_gt_mask, brand_ok)."""
    tp = [False] * len(preds)
    matched_gt = [False] * len(gt_boxes)
    brand_ok = [False] * len(preds)
    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for j, gtb in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            v = iou(pred["bbox"], gtb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp[i] = True
            brand_ok[i] = _brands_agree(pred.get("brand"), gt_brands[best_j])
            matched_gt[best_j] = True
    return tp, matched_gt, brand_ok


def _resolve_brand(x) -> str:
    return (x or "").upper().strip()


def _brands_agree(pred_brand, gt_brand) -> bool:
    """True when a TP's brand attribution matches the GT brand."""
    if not pred_brand or not gt_brand:
        return False
    return _resolve_brand(pred_brand) == _resolve_brand(gt_brand)


def evaluate_image(
    gt_detections: List[dict],
    predictions: List[dict],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Per-image matching stats used to accumulate dataset-wide metrics.
    """
    gt_boxes = [g["bbox"] for g in gt_detections]
    gt_brands = [g.get("brand", "") for g in gt_detections]
    tp, matched, brand_ok = _match_predictions(gt_boxes, gt_brands, predictions, iou_thresh)
    return {
        "gt_count": len(gt_detections),
        "pred_count": len(predictions),
        "tp": sum(tp),
        "fp": len(predictions) - sum(tp),
        "fn": len(gt_detections) - sum(matched),
        "brand_correct": sum(brand_ok),
        "confidences": [p.get("confidence", 0.0) for p in predictions],
        "tp_flags": tp,
    }


def pr_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def accumulate(
    per_image: List[Dict],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Aggregate per-image matching stats into dataset metrics.

    Uses the standard matching protocol: sort all predictions by confidence
    across the whole dataset, walk the (unique) ranks, accumulate TP/FP and
    GT count to build a global PR curve, then integrate for mAP@0.5 and report
    P/R/F1 at the default operating point (conf >= 0.3).
    """
    tp = sum(i["tp"] for i in per_image)
    fp = sum(i["fp"] for i in per_image)
    fn = sum(i["fn"] for i in per_image)
    op = pr_f1(tp, fp, fn)
    op["tp"], op["fp"], op["fn"] = tp, fp, fn
    op["n_images"] = len(per_image)
    brand_correct = sum(i["brand_correct"] for i in per_image)
    op["brand_correct"] = brand_correct
    op["brand_accuracy"] = round(brand_correct / tp, 4) if tp else 0.0

    # ── mAP@0.5 via confidence-sorted global PR curve ──
    confs = []
    for img in per_image:
        for c, flag in zip(img["confidences"], img["tp_flags"]):
            confs.append((c, flag))
    total_gt = tp + fn

    ap = 0.0
    if confs and total_gt > 0:
        # Sort descending by confidence, stable on ties.
        confs.sort(key=lambda x: -x[0])
        # Rank-based: accumulate at each unique confidence step.
        prev_c = None
        running_tp = 0
        running_fp = 0
        precisions: List[float] = []
        recalls: List[float] = []
        for c, flag in confs:
            if flag:
                running_tp += 1
            else:
                running_fp += 1
            if c != prev_c:
                p = running_tp / (running_tp + running_fp) if (running_tp + running_fp) else 0.0
                r = running_tp / total_gt if total_gt else 0.0
                precisions.append(p)
                recalls.append(r)
                prev_c = c
        # Numerically stable AP: integrate the monotonically-decreasing
        # precision envelope over recall (VOC-style area under PR curve).
        ap = 0.0
        max_p = 0.0
        prev_r = 0.0
        for r, p in sorted(zip(recalls, precisions), key=lambda x: x[0]):
            max_p = max(max_p, p)
            ap += max_p * (r - prev_r)
            prev_r = r

    op["map50"] = float(ap)
    return op


def evaluate(
    images: List[Dict],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Evaluate a list of images, each with 'gt' and 'pred' keys.

    All predictions are matched (each backend applies its own detection
    threshold before returning) so the confidence-sweep PR curve is complete;
    P/R/F1 are the operating-point numbers at the backend's own threshold.
    """
    per_image = [
        evaluate_image(img.get("gt", []), img.get("pred", []), iou_thresh)
        for img in images
    ]

    agg = accumulate(per_image, iou_thresh)
    agg["iou_threshold"] = iou_thresh
    agg["conf_threshold"] = "backend_default"
    return {"aggregate": agg, "per_image": per_image}
