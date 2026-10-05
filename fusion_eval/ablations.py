"""Modality ablation: re-run fusion with families disabled and measure deltas.

Ablations reuse the same cached per-entry predictions (detection is fixed);
they re-run ONLY the fusion verdict with `disable_families` set, so deltas are
purely attributable to modality ownership, not model jitter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .metrics import compute_metrics

COMBOS = [
    ("ALL", []),
    ("NO_LOGO", ["logo"]),
    ("NO_OCR", ["ocr"]),
    ("NO_ASR", ["speech"]),
    ("NO_PRODUCT", ["product"]),
    ("VISUAL_ONLY", ["speech", "audio"]),
    ("TEXT_ONLY", ["logo", "product", "audio"]),
]


def _filter_ledger(ledger: Dict[str, List[dict]], disable: List[str]) -> Dict[str, List[dict]]:
    return {
        b: [it for it in items if it["family"] not in disable]
        for b, items in ledger.items()
        if any(it["family"] not in disable for it in items)
    }


def rerun_fusion(
    pred: dict,
    disable: List[str],
    *,
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 1.0,
    time_bucket: float = 2.0,
    accept: float = 0.30,
    margin_min: float = 0.15,
    agreement_bonus: float = 0.30,
    contradiction_penalty: float = 0.5,
) -> dict:
    """Re-fuse a cached prediction row with `disable` families removed.

    The row must carry `_ledger` (kept by the pipeline for exactly this purpose).
    Returns a new prediction row; detection is never re-run.
    """
    from src.layer2.evidence_fusion import fuse_candidates

    ledger = _filter_ledger(pred.get("_ledger") or {}, disable)
    f = fuse_candidates(
        ledger,
        base_weights=weights or None,
        temperature=temperature,
        time_bucket=time_bucket,
        accept=accept,
        margin_min=margin_min,
        agreement_bonus=agreement_bonus,
        contradiction_penalty=contradiction_penalty,
    )
    ranking = f.get("ranking") or []
    return {
        **pred,
        "_ledger": ledger,
        "winner": f.get("winner"),
        "prob": ranking[0]["prob"] if ranking else 0.0,
        "margin": f.get("margin", 0.0),
        "verdict": f.get("verdict"),
        "families": ranking[0].get("families", {}) if ranking else {},
        "top3": [r["candidate"] for r in ranking[:3]],
    }


def ablation_report(base_preds: List[dict], per_combo: Dict[str, List[dict]]) -> dict:
    base = compute_metrics(base_preds)
    report = {"base": base, "combos": {}}
    for name, rows in per_combo.items():
        m = compute_metrics(rows)
        report["combos"][name] = {
            "top1": m["top1_accuracy"],
            "top1_covered": m["top1_accuracy_covered"],
            "f1": m["f1"],
            "ece": m["ece"],
            "abstention": m["abstention_rate"],
            "delta_top1_vs_all": round(m["top1_accuracy"] - base["top1_accuracy"], 4),
            "delta_f1_vs_all": round(m["f1"] - base["f1"], 4),
            "delta_ece_vs_all": round(m["ece"] - base["ece"], 4),
        }
    return report
