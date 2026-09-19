"""Temperature calibration: grid-search T minimising ECE, then report the curve.

TODO(calibration-set): refit T on a held-out labeled set once it exists. Here we
demonstrate the mechanism on whatever covered predictions the current run yields.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .reliability import reliability_curve


def best_temperature(preds: List[dict], search=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)) -> dict:
    probs = np.asarray([r["prob"] for r in preds], dtype=float)
    y = np.asarray([r["winner"] == r["gt_brand"] for r in preds], dtype=float)
    res = []
    for t in search:
        scaled = probs ** (1.0 / t)
        _, ece, brier = reliability_curve(list(scaled), list(y.astype(bool)))
        res.append((t, ece, brier))
    res.sort(key=lambda r: r[1])
    return {
        "temperature": res[0][0],
        "ece": round(res[0][1], 4),
        "brier": round(res[0][2], 4),
        "ece_all": {str(t): round(e, 4) for t, e, _ in res},
    }