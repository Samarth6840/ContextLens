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

    probs outside [0,1] (or NaN) match no bin, so they would inflate the
    denominator `p.size` without ever contributing to the numerator — silently
    deflating ECE. Raise instead: a miscalibrated metric is worse than none.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.size == 0:
        return [], 0.0, 0.0
    if p.size != y.size:
        raise ValueError(f"probs/correct length mismatch: {p.size} vs {y.size}")
    bad = ~np.isfinite(p) | (p < 0.0) | (p > 1.0)
    if bad.any():
        raise ValueError(
            f"probabilities must be finite and in [0,1]; got "
            f"{int(bad.sum())} bad value(s), e.g. {p[bad][:3].tolist()}"
        )
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
    # Proper binned ECE: sample-weighted mean gap between confidence and
    # accuracy WITHIN each bin. The previous mean(abs(p - y)) was mean absolute
    # error, not ECE — it punishes confident-correct predictions and ignores
    # calibration entirely (a model at 0.9 on 90%-accurate data scores ~0.1 MAE
    # instead of ~0).
    ece = sum(b["n"] * abs(b["avg_prob"] - b["accuracy"]) for b in bins) / p.size
    brier = float(np.mean((p - y) ** 2))
    return bins, float(ece), brier