"""Rule-based failure taxonomy. Every wrong/abstained call gets one label so
the biggest failure category is obvious before any model tweaking.

FP_* (wrong accept) / FN_* (missed) labels below mirror the design brief.
"""

from __future__ import annotations

from typing import Any, Dict

from .metrics import ABSTAIN_VERDICTS


def classify(pred: dict) -> str:
    winner, gt = pred.get("winner"), pred.get("gt_brand")
    verdict, diff = pred.get("verdict"), pred.get("difficulty")
    fam = pred.get("families", {}) if isinstance(pred.get("families"), dict) else {}

    # A declined row has no answer, whatever winner the fusion layer reported.
    if verdict in ABSTAIN_VERDICTS:
        # Declining a NEGATIVE is the correct outcome, not a miss. Labelling it
        # FN_* made correct restraint look like the dominant failure mode.
        if gt is None or not pred.get("visible"):
            return "OK"
        return "FN_LOW_EVIDENCE" if verdict == "low_support" else "FN_UNKNOWN"

    # A row whose winner equals ground truth is not a failure, whatever the
    # verdict said about confidence. An `ambiguous` verdict with the right
    # winner is an under-confident success, not a false positive.
    if winner == gt:
        return "OK"

    if winner is None or not pred.get("visible"):
        return "FP_UNKNOWN_BRAND"
    if verdict == "ambiguous":
        return "FP_AMBIGUOUS_RIVAL"
    if diff == "small" and fam.get("logo", 0) < 0.4:
        return "FN_SMALL_LOGO"
    if diff == "blur":
        return "FN_BLUR"
    return "FP_WRONG_CANDIDATE"


def _families(pred: dict) -> Dict[str, Any]:
    fam = pred.get("families", {})
    return fam if isinstance(fam, dict) else {}