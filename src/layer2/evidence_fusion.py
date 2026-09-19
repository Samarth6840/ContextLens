"""Candidate-centric, dependency-aware evidence fusion.

Replaces "sum of fixed weights -> threshold" with:

    evidence -> quality/reliability -> temporal alignment ->
    dependency discount -> agreement bonus -> candidate score ->
    conflict/verdict (accept / ambiguous / abstain) -> calibrated prob

Current reliability is signal-quality heuristics (resolution confidence,
OCR confidence ceiling, top-k product consistency). A LEARNED reliability /
LightGBM fusion model is the designed upgrade path and is deferred until a
labeled candidate-evidence dataset exists (see ponytail: notes below).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Legacy modality base weights (kept as the contribution priors; effective
# contribution = base_weight * reliability * temporal persistence).
BASE_WEIGHTS = {
    "logo": 0.30,
    "speech": 0.20,
    "ocr": 0.18,
    "product": 0.18,
    "audio": 0.10,
}

_QUALITY_FLOOR = 0.30   # clamped minimum reliability multiplier
_QUALITY_CEIL = 1.0


def _effective(strength: float, quality: Optional[float]) -> float:
    """Reliability-scaled strength. quality is the modality quality estimate
    (resolution tier / OCR confidence ceiling / similarity); model confidence
    alone is NOT treated as reliability, per the design brief."""
    rel = _QUALITY_CLAMP(quality) if quality is not None else 1.0
    return float(strength) * rel


def _QUALITY_CLAMP(q: float) -> float:
    return max(_QUALITY_FLOOR, min(_QUALITY_CEIL, float(q)))


def _bucket(ts: Optional[float], frame_index, time_bucket: float) -> int:
    """Correlation bucket: evidence in the same time window is ONE dependent
    vote per family (dependency discount), not N independent votes."""
    if ts is None:
        return -1
    return int(ts // time_bucket) if time_bucket > 0 else 0


def _family_scores(items: List[dict], time_bucket: float) -> Dict[str, float]:
    fam: Dict[str, Dict[int, float]] = {}
    fam_has_time: Dict[str, bool] = {}
    for it in items:
        f = it["family"]
        eff = _effective(it.get("strength", 0.0), it.get("quality"))
        b = _bucket(it.get("timestamp"), it.get("frame_index"), time_bucket)
        bucket_map = fam.setdefault(f, {})
        bucket_map[b] = max(bucket_map.get(b, 0.0), eff)
        fam_has_time[f] = fam_has_time.get(f, False) or it.get("timestamp") is not None

    out: Dict[str, float] = {}
    for f, buckets in fam.items():
        top = max(buckets.values())
        if fam_has_time[f] and len(buckets) > 1:
            # Temporal persistence: the same family reinforcing the candidate
            # across DISTINCT moments adds modest strength (n/(n+2) saturate).
            persist = len(buckets) / (len(buckets) + 2.0)
            out[f] = top * (1.0 + 0.25 * persist)
        else:
            out[f] = top
    return out


def _agreement(items: List[dict], time_bucket: float) -> float:
    """Cross-modal temporal agreement: fraction of family PAIRS that co-occur
    within a time bucket. Independent modalities agreeing in time is the
    strongest signal (logo + OCR + speech in the same window)."""
    fam_ts: Dict[str, List[float]] = {}
    for it in items:
        ts = it.get("timestamp")
        if ts is not None:
            fam_ts.setdefault(it["family"], []).append(float(ts))
    overlap_pairs = 0
    fams = sorted(fam_ts)
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            a, b = fams[i], fams[j]
            ta, tb = fam_ts[a], fam_ts[b]
            if any(
                abs(x - y) <= time_bucket for x in ta for y in tb
            ):
                overlap_pairs += 1
    total_pairs = len(fams) * (len(fams) - 1) / 2.0
    if total_pairs <= 0:
        return 0.0
    return min(1.0, overlap_pairs / total_pairs)


def fuse_candidates(
    ledger: Dict[str, List[dict]],
    *,
    base_weights: Optional[Dict[str, float]] = None,
    temperature: float = 1.0,
    time_bucket: float = 2.0,
    accept: float = 0.30,
    margin_min: float = 0.15,
    agreement_bonus: float = 0.30,
) -> dict:
    """Fuse per-candidate evidence into a ranked, calibrated verdict.

    ledger: {brand: [{family, strength, quality?, timestamp?, frame_index?}, ...]}

    Returns:
        ranking  : [{candidate, score, prob, families}...]  (descending prob)
        winner   : top candidate or None
        margin   : top - second prob (top if a single candidate)
        conflict : 1 - min(margin, 1)  (0 = none, 1 = total)
        verdict  : confident | ambiguous | low_support | abstain
        reason   : human-readable explanation
        temperature: applied scaling factor (1.0 = identity)
    """
    weights = dict(base_weights or BASE_WEIGHTS)
    scores: List[dict] = []
    for brand, items in ledger.items():
        fam = _family_scores(items, time_bucket)
        active_sum = sum(weights[f] for f in fam if f in weights)
        if active_sum <= 0:
            continue
        agree = _agreement(items, time_bucket)
        raw = sum(weights.get(f, 0.0) * s for f, s in fam.items())
        raw *= 1.0 + agreement_bonus * agree
        prob = max(0.0, min(1.0, raw / active_sum))
        if temperature > 0:
            prob = prob ** (1.0 / temperature)
        scores.append(
            {
                "candidate": brand,
                "score": round(raw, 4),
                "prob": round(prob, 4),
                "families": {f: round(s, 4) for f, s in fam.items()},
                "temporal_agreement": round(agree, 4),
                "supporting": [
                    {"family": it["family"],
                     "strength": round(float(it.get("strength", 0.0)), 4),
                     "quality": round(float(it["quality"]), 4) if it.get("quality") is not None else None,
                     "timestamp": it.get("timestamp"),
                     "frame_index": it.get("frame_index")}
                    for it in sorted(items, key=lambda e: -float(e.get("strength", 0.0)))[:8]
                ],
            }
        )

    if not scores:
        return {
            "ranking": [],
            "winner": None,
            "margin": 0.0,
            "conflict": 0.0,
            "verdict": "abstain",
            "reason": "no candidate evidence across any modality",
            "temperature": temperature,
        }

    scores.sort(key=lambda s: s["prob"], reverse=True)
    top, second = scores[0], (scores[1] if len(scores) > 1 else None)
    margin = top["prob"] - (second["prob"] if second else 0.0)
    conflict = round(1.0 - min(1.0, margin), 4)

    if top["prob"] < accept:
        verdict, reason = "low_support", f"top candidate {top['candidate']} below accept threshold ({top['prob']} < {accept})"
    elif second and margin < margin_min:
        verdict, reason = "ambiguous", (
            f"top candidates {top['candidate']} ({top['prob']}) vs "
            f"{second['candidate']} ({second['prob']}) too close (margin {margin:.3f} < {margin_min})"
        )
    else:
        second_note = f" vs {second['candidate']} ({second['prob']})" if second else ""
        verdict, reason = "confident", (
            f"{top['candidate']} leads at {top['prob']}{second_note} (margin {margin:.3f})"
        )

    return {
        "ranking": scores,
        "winner": top["candidate"],
        "margin": round(margin, 4),
        "conflict": conflict,
        "verdict": verdict,
        "reason": reason,
        "temperature": temperature,
    }


def _check() -> None:
    # Dependency discount: two OCR reads of the SAME brand in the SAME bucket
    # are one vote, not two. logo at t=0 must equal logo+logo at t=0.
    solo = [{"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.0}]
    dup = [
        {"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.0},
        {"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.4},
    ]
    assert fuse_candidates({"N": solo})["ranking"][0]["families"]["logo"] == \
           fuse_candidates({"N": dup})["ranking"][0]["families"]["logo"], "dependency discount"

    # Agreement bonus: logo + OCR in the same window beats logo alone.
    two = [*solo, {"family": "ocr", "strength": 0.9, "quality": 1.0, "timestamp": 0.5}]
    assert fuse_candidates({"N": two})["ranking"][0]["prob"] > \
           fuse_candidates({"N": solo})["ranking"][0]["prob"], "temporal agreement bonus"

    # Conflict detection: close margin -> ambiguous.
    close = fuse_candidates({"N": solo, "A": [{"family": "logo", "strength": 0.88, "quality": 1.0, "timestamp": 0.0}]})
    assert close["verdict"] == "ambiguous", close

    # Abstain: empty ledger.
    assert fuse_candidates({})["verdict"] == "abstain"

    # Reliability: low quality shrinks the score.
    weak = [{"family": "logo", "strength": 0.9, "quality": 0.15, "timestamp": 0.0}]
    assert fuse_candidates({"N": weak})["ranking"][0]["families"]["logo"] < 0.9
    print("evidence_fusion self-check OK")


if __name__ == "__main__":
    _check()