"""Binned reliability curve: predicted probability vs empirical accuracy."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def reliability_curve(
    probs: List[float], correct: List[bool], n_bins: int = 10
) -> Tuple[List[dict], float, float]:
    """Binned reliability + ECE + Brier.

    probs[i] in [0,1] is the winner's fused probability; correct[i] is whether
    that winner matched ground truth. Only covered (non-abstained) samples.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.size == 0:
        return [], 0.0, 0.0
    bins = []
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        bins.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "n": n,
            "avg_prob": round(float(p[m].mean()) if n else 0.0, 4),
            "accuracy": round(float(y[m].mean()) if n else 0.0, 4),
        })
    ece = float(np.mean(np.abs(p - y)))
    brier = float(np.mean((p - y) ** 2))
    return bins, ece, brier