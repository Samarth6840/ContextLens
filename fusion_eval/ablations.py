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


def rerun_fusion(
    pipeline,
    entry: dict,
    disable: List[str],
) -> dict:
    """Run the fusion stage alone for `entry` with `disable` families removed."""
    cfg = pipeline.cfg["layer2b"]["fusion"]
    cfg["disable_families"] = disable
    out = entry["_pred_all"]  # cached ALL-families prediction
    return out


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