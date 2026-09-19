"""Rule-based failure taxonomy. Every wrong/abstained call gets one label so
the biggest failure category is obvious before any model tweaking.

FP_* (wrong accept) / FN_* (missed) labels below mirror the design brief.
"""

from __future__ import annotations

from typing import Any, Dict

def classify(pred: dict) -> str:
    winner, gt = pred.get("winner"), pred.get("gt_brand")
    verdict, diff = pred.get("verdict"), pred.get("difficulty")
    fam = pred.get("families", {}) if isinstance(pred.get("families"), dict) else {}

    if verdict == "abstain":
        return "FN_UNKNOWN" if diff == "negative" else "FN_UNKNOWN"
    if verdict == "low_support":
        return "FN_LOW_EVIDENCE"

    if winner != gt:
        if winner is None or not pred.get("visible"):
            return "FP_UNKNOWN_BRAND"
        if verdict == "ambiguous":
            return "FP_AMBIGUOUS_RIVAL"
        if diff == "small" and fam.get("logo", 0) < 0.4:
            return "FN_SMALL_LOGO"
        if diff == "blur":
            return "FN_BLUR"
        if not fam:  # nothing fired for the winner
            return "FP_WRONG_CANDIDATE"
        if gt and winner and fam and set(winner).union({gt}):
            return "FP_WRONG_CANDIDATE"
        return "FP_WRONG_CANDIDATE"

    # correct winner but covered-with-ambiguity or near-miss is still a flag
    if verdict == "ambiguous" and winner == gt:
        return "FP_CORRELATED_EVIDENCE" if pred.get("margin", 0) < 0.1 else "FP_TEMPORAL"

    return "OK"


def _families(pred: dict) -> Dict[str, Any]:
    fam = pred.get("families", {})
    return fam if isinstance(fam, dict) else {}