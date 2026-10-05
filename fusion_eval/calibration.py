"""Temperature calibration: grid-search T minimising binned ECE on a fit split,
then report the curve on a held-out split.

Fitting T and reporting ECE on the SAME rows measures how well the grid search
memorised that set, not how well the calibrated model generalises. Pass
`test_rows` to get an honest number; the returned `ece`/`brier` are then measured
on rows the temperature was never chosen against.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .reliability import reliability_curve


def best_temperature(
    preds: List[dict],
    search=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    test_rows: Optional[List[dict]] = None,
) -> dict:
    """Pick T minimising binned ECE on `preds`; report on `test_rows` if given."""
    probs = np.asarray([r["prob"] for r in preds], dtype=float)
    y = np.asarray([r["winner"] == r["gt_brand"] for r in preds], dtype=float)
    res = []
    for t in search:
        scaled = probs ** (1.0 / t)
        _, ece, brier = reliability_curve(list(scaled), list(y.astype(bool)))
        res.append((t, ece, brier))
    res.sort(key=lambda r: r[1])
    best_t = res[0][0]
    out = {
        "temperature": best_t,
        "ece": round(res[0][1], 4),
        "brier": round(res[0][2], 4),
        "ece_all": {str(t): round(e, 4) for t, e, _ in res},
        "n_fit": len(preds),
    }
    if test_rows:
        tp = np.asarray([r["prob"] for r in test_rows], dtype=float) ** (1.0 / best_t)
        ty = np.asarray([r["winner"] == r["gt_brand"] for r in test_rows], dtype=bool)
        _, tece, tbrier = reliability_curve(list(tp), list(ty))
        out["heldout_ece"] = round(tece, 4)
        out["heldout_brier"] = round(tbrier, 4)
        out["n_test"] = len(test_rows)
    return out