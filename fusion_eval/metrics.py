"""Evaluation metrics over pipeline predictions.

Conventions
-----------
- covered   = verdict in {confident, ambiguous}     (system committed)
- abstained = verdict in {abstain, low_support}     (system declined)
- accepted  = verdict == confident                  (the only AUTO-ACCEPT state)
- positive  = ground truth has a visible brand
- negative  = ground truth brand is None
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .reliability import reliability_curve

ABSTAIN_VERDICTS = {"abstain", "low_support"}
COVERED_VERDICTS = {"confident", "ambiguous"}


def compute_metrics(preds: List[dict]) -> Dict[str, float]:
    """Compute the metric ledger from a list of prediction rows.

    Each row: {video_id, gt_brand, visible, winner, prob, verdict,
               top3(list), latency_layer1, difficulty}

    Rows whose verdict is in ABSTAIN_VERDICTS are scored as having no answer,
    regardless of the `winner` the fusion layer reports for them.
    """
    n = len(preds)
    out: Dict[str, float] = {}

    # A declined row has NO answer. fuse_candidates() still returns its top
    # candidate as `winner` for `low_support`, so scoring rows as-is made a
    # correct decline on a negative count as a wrong answer — the system was
    # penalised for the behaviour we asked of it. Strip the winner on abstained
    # rows so a decline is credited on a negative (gt None) and still scores as
    # a miss on a positive (gt a real brand).
    rows = [
        {**p, "winner": None} if p.get("verdict") in ABSTAIN_VERDICTS else p
        for p in preds
    ]

    covered = [p for p in rows if p["verdict"] in COVERED_VERDICTS]
    accepted = [p for p in rows if p["verdict"] == "confident"]
    positives = [p for p in rows if p.get("visible")]
    negatives = [p for p in rows if not p.get("visible")]

    def hits(group):
        return sum(1 for r in group if r.get("winner") == r.get("gt_brand"))

    acc_all = hits(rows) / n if n else 0.0
    acc_covered = hits(covered) / len(covered) if covered else 0.0
    acc_top3_covered = (
        sum(1 for r in covered if r.get("gt_brand") in (r.get("top3") or []))
        / len(covered) if covered else 0.0
    )

    tp = sum(1 for r in accepted if r.get("winner") == r.get("gt_brand"))
    fp = sum(1 for r in accepted if r.get("winner") != r.get("gt_brand"))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    # every positive is either a TP or an FN, so len(positives) == tp + fn
    rec = tp / max(1, len(positives))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    fp_claims = sum(1 for r in accepted if r.get("visible") is False)
    fpr = fp_claims / len(negatives) if negatives else 0.0     # negative entries wrongly accepted

    abst = sum(1 for p in rows if p["verdict"] in ABSTAIN_VERDICTS) / n if n else 0.0
    amb = sum(1 for p in rows if p["verdict"] == "ambiguous") / n if n else 0.0

    _, ece, brier = reliability_curve(
        [r["prob"] for r in covered], [r["winner"] == r["gt_brand"] for r in covered]
    )

    lat = [r.get("latency_layer1") or 0.0 for r in rows]
    mem = [r.get("gpu_mem_mb") or 0.0 for r in rows]

    out.update({
        "n": float(n),
        "top1_accuracy": round(acc_all, 4),
        "top1_accuracy_covered": round(acc_covered, 4),
        "top3_accuracy_covered": round(acc_top3_covered, 4),
        "precision_accepted": round(prec, 4),
        "recall_positive": round(rec, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "abstention_rate": round(abst, 4),
        "ambiguous_rate": round(amb, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "latency_layer1_mean_s": round(float(np.mean(lat)) if lat else 0.0, 3),
        "gpu_mem_mean_mb": round(float(np.mean(mem)) if mem else 0.0, 1),
        # Which Layer-2a weighting path actually ran per row. The
        # "learned vs heuristic" claim is only meaningful if learned_gating
        # dominates; a large fallback count means the gate is being overridden.
        "weight_source_counts": dict(Counter(
            str(p.get("weight_source", "unknown")) for p in preds)),
    })
    return out


def split_by_difficulty(preds: List[dict]) -> Dict[str, Dict[str, float]]:
    return split_by_field(preds, "difficulty")


def split_by_field(preds: List[dict], key: str) -> Dict[str, Dict[str, float]]:
    """Metric ledger grouped by any row field (e.g. difficulty, length)."""
    per: Dict[str, List[dict]] = {}
    for p in preds:
        per.setdefault(str(p.get(key, "?")), []).append(p)
    return {k: compute_metrics(v) for k, v in sorted(per.items())}


def split_by_modality(preds: List[dict]) -> Dict[str, Dict[str, float]]:
    """Metric ledger grouped by which modalities carried evidence.

    Answers the brief's "accuracy vs modality availability" question: does the
    system still work when only visual, only text, or all modalities are
    present? Rows expose a `modalities` dict ({logo, ocr, speech, product,
    audio_event}); missing data is treated as "unknown" so old prediction
    files still evaluate.
    """
    per: Dict[str, List[dict]] = {}
    for p in preds:
        mods = p.get("modalities")
        if not isinstance(mods, dict):
            label = "unknown"
        else:
            active = [k for k, v in mods.items() if v]
            label = "+".join(sorted(active)) if active else "none"
        per.setdefault(label, []).append(p)
    return {k: compute_metrics(v) for k, v in sorted(per.items())}