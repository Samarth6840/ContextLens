"""Candidate-centric, dependency-aware multimodal evidence aggregation.

This is multimodal evidence aggregation + candidate resolution, not a learned
fused model: it combines quality-weighted modality evidence with temporal
grounding, dependency/track discounts, negative (contradiction) evidence and
agreement bonuses into a ranked verdict. A LEARNED fusion model is the designed
deferred upgrade (see the training-data note below).

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

# Default weight applied to a candidate's contradiction score when computing
# net evidence. Kept below 1.0 so an explicit rival sighting can erode but not
# single-handedly erase strong supporting evidence for the candidate.
_CONTRADICTION_PENALTY = 0.5

# Evidence polarity keys. Items default to "support" so existing ledgers keep
# their exact behavior; "contradict" marks evidence that argues AGAINST the
# candidate it is filed under (e.g. product retrieval naming a rival brand, or
# OCR reading competitor text inside a box resolved to this brand).
SUPPORT = "support"
CONTRADICT = "contradict"


def _polarity(item: dict) -> str:
    return str(item.get("polarity") or SUPPORT)


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


def _fam_persistence(fam_persist: Dict[str, int], n: int) -> float:
    """Saturating persistence value for a track/frame count (n/(n+2))."""
    return n / (n + 2.0) if n > 0 else 0.0


def _family_scores(items: List[dict], time_bucket: float) -> Dict[str, float]:
    fam: Dict[str, Dict[int, float]] = {}
    fam_has_time: Dict[str, bool] = {}
    # Explicit track persistence: when a physical-object track (see
    # src/layer2/evidence_tracking.py) survives n frames, its evidence is
    # stronger than an isolated single-frame sighting. Kept separate from the
    # bucket-persistence below so a track that fires once per bucket still gets
    # credit for its full extent.
    fam_persist: Dict[str, int] = {}
    for it in items:
        f = it["family"]
        eff = _effective(it.get("strength", 0.0), it.get("quality"))
        b = _bucket(it.get("timestamp"), it.get("frame_index"), time_bucket)
        bucket_map = fam.setdefault(f, {})
        bucket_map[b] = max(bucket_map.get(b, 0.0), eff)
        fam_has_time[f] = fam_has_time.get(f, False) or it.get("timestamp") is not None
        p = int(it.get("persistence") or 0)
        if p > fam_persist.get(f, 0):
            fam_persist[f] = p

    out: Dict[str, float] = {}
    for f, buckets in fam.items():
        top = max(buckets.values())
        frac = 0.0
        if fam_has_time[f] and len(buckets) > 1:
            # Temporal persistence: the same family reinforcing the candidate
            # across DISTINCT moments adds modest strength (n/(n+2) saturate).
            frac = len(buckets) / (len(buckets) + 2.0)
        frac = max(frac, _fam_persistence(fam_persist, fam_persist.get(f, 0)))
        out[f] = top * (1.0 + 0.25 * frac) if frac > 0 else top
    return out


# Families that name a brand from a visual source, and the subset that reads a
# wordmark/printed text identity (vs. recognizing a product by appearance).
VISUAL_FAMILIES = {"logo", "ocr", "product"}
WORDMARK_FAMILIES = {"logo", "ocr"}


def _support_families(items: List[dict]) -> set:
    return {it["family"] for it in items if _polarity(it) != CONTRADICT}


def _contradict_item(src: dict, source_brand: str, rule: str) -> dict:
    """A copy of `src` re-filed as contradicting evidence under another brand."""
    return {
        "family": src.get("family"),
        "strength": float(src.get("strength", 0.0)),
        "quality": src.get("quality"),
        "timestamp": src.get("timestamp"),
        "frame_index": src.get("frame_index"),
        "polarity": CONTRADICT,
        "source_brand": source_brand,
        "contradiction_rule": rule,
    }


def derive_cross_modal_contradictions(
    ledger: Dict[str, List[dict]],
    *,
    min_product_similarity: float = 0.6,
    min_visual_strength: float = 0.5,
) -> Dict[str, List[dict]]:
    """Mine negative evidence from CROSS-MODAL conflicts in the ledger.

    Rival evidence normally only raises the rival's own score (the margin then
    decides). These rules additionally file a rival's strong claim as
    contradicting evidence against a candidate it disagrees with, so a genuine
    modality conflict erodes the competing score instead of being ignored:

      * product_vs_logo — a candidate is established by a WORDMARK (logo/OCR)
        but the product classifier strongly names a different, wordmark-less
        brand: two visual classifiers disagree about the same content.
      * asr_vs_visual — a candidate is visually established and unheard in the
        audio, while another brand appears ONLY in speech: audio and vision
        each name a different brand.
      * visual_vs_asr — the inverse: a candidate rests on audio alone while a
        different brand has visual proof.

    Rules are deliberately scoped to genuine one-sided conflicts (e.g. a rival
    seen in BOTH audio and video is consistent, not a contradiction) so a
    multi-brand video never self-penalises. Returns a new ledger; the input is
    left untouched.
    """
    out: Dict[str, List[dict]] = {b: list(items) for b, items in ledger.items()}
    fams = {b: _support_families(items) for b, items in out.items()}

    for a, fams_a in fams.items():
        wordmark_a = bool(WORDMARK_FAMILIES & fams_a)
        visual_a = bool(VISUAL_FAMILIES & fams_a)
        for b, fams_b in fams.items():
            if a == b:
                continue
            items_b = [
                it for it in out[b] if _polarity(it) != CONTRADICT
            ]
            wordmark_b = bool(WORDMARK_FAMILIES & fams_b)
            visual_b = bool(VISUAL_FAMILIES & fams_b)

            # 1. Wordmark identity vs product-classifier identity.
            if wordmark_a and "product" not in fams_a and not wordmark_b:
                for it in items_b:
                    if it["family"] == "product" and float(it.get("strength", 0.0)) >= min_product_similarity:
                        out[a].append(_contradict_item(it, b, "product_vs_logo"))

            # 2. Visually established, but another brand is speech-only.
            if visual_a and "speech" not in fams_a and "speech" in fams_b and not visual_b:
                for it in items_b:
                    if it["family"] == "speech":
                        out[a].append(_contradict_item(it, b, "asr_vs_visual"))

            # 3. Audio-only candidate vs another brand's visual proof.
            if "speech" in fams_a and not visual_a and visual_b and "speech" not in fams_b:
                for it in items_b:
                    if it["family"] in VISUAL_FAMILIES and _effective(
                        it.get("strength", 0.0), it.get("quality")
                    ) >= min_visual_strength:
                        out[a].append(_contradict_item(it, b, "visual_vs_asr"))

    return out


def _collapse_tracks(
    items: List[dict], time_bucket: float
) -> List[dict]:
    """Collapse repeated observations of ONE physical track into one item.

    Dependency discount already merges same-family evidence inside a 2s bucket;
    a track (see src/layer2/evidence_tracking.py) additionally merges the SAME
    object seen across distant buckets, which would otherwise look like N
    independent corroborating votes. Each track becomes ONE item carrying its
    best observation as strength, its mean quality, and its frame count as
    `persistence`. Items without a track_id pass through untouched, so ledgers
    that never ran tracking behave exactly as before.
    """
    out: List[dict] = []
    by_track: Dict[tuple, dict] = {}
    order: List[dict] = []
    for it in items:
        tid = it.get("track_id")
        if tid is None:
            out.append(it)
            continue
        key = (it["family"], tid, _polarity(it))
        entry = by_track.get(key)
        if entry is None:
            entry = {
                "family": it["family"],
                "polarity": _polarity(it),
                "track_id": tid,
                "strength": 0.0,
                "_qualities": [],
                "best_timestamp": it.get("timestamp"),
                "best_strength": -1.0,
                "frame_index": it.get("frame_index"),
                "_frames": 0,
            }
            by_track[key] = entry
            order.append(entry)
        s = float(it.get("strength", 0.0))
        entry["_frames"] += 1
        entry["_qualities"].append(it.get("quality"))
        if s > entry["best_strength"]:
            entry["best_strength"] = s
            entry["strength"] = s
            entry["best_timestamp"] = it.get("timestamp")
            entry["frame_index"] = it.get("frame_index")
    for entry in order:
        quals = [q for q in entry.pop("_qualities") if q is not None]
        entry["quality"] = sum(quals) / len(quals) if quals else None
        entry["persistence"] = entry.pop("_frames")
        entry["timestamp"] = entry.pop("best_timestamp")
        entry.pop("best_strength")
        out.append(entry)
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
    contradiction_penalty: float = _CONTRADICTION_PENALTY,
    collapse_tracks: bool = True,
) -> dict:
    """Fuse per-candidate evidence into a ranked, calibrated verdict.

    ledger: {brand: [{family, strength, quality?, timestamp?, frame_index?,
                      polarity?, track_id?, persistence?}, ...]}

    polarity (default "support") separates evidence FOR a candidate from
    evidence AGAINST it ("contradict"), so a candidate accumulates both a
    `support_score` and a `contradiction_score` and is ranked on
    `net_evidence = support - contradiction_penalty * contradiction`.

    Absence is NOT negative evidence: a candidate with no items at all is
    "unknown" (never penalised), whereas explicit contradicting items make it
    "negative ". This distinction is what stops a missing detection from
    becoming a silent false penalty.

    Returns:
        ranking  : [{candidate, score, prob, families, support_score,
                     contradiction_score, net_evidence, evidence_state}...]
        winner   : top candidate or None
        margin   : top - second prob (top if a single candidate)
        conflict : 1 - min(margin, 1)  (0 = none, 1 = total)
        verdict  : confident | ambiguous | low_support | abstain
        negative_evidence: True when any candidate carries contradicting items
        reason   : human-readable explanation
        temperature: applied scaling factor (1.0 = identity)
    """
    weights = dict(base_weights or BASE_WEIGHTS)
    scores: List[dict] = []
    for brand, items in ledger.items():
        if collapse_tracks:
            items = _collapse_tracks(items, time_bucket)
        support_items = [it for it in items if _polarity(it) != CONTRADICT]
        contradict_items = [it for it in items if _polarity(it) == CONTRADICT]
        fam = _family_scores(support_items, time_bucket)
        active_sum = sum(weights[f] for f in fam if f in weights)
        if active_sum <= 0:
            # No supported family at all — the candidate is either absent
            # (unknown) or only contradicted (negative). It is never ranked.
            continue
        agree = _agreement(support_items, time_bucket)
        raw = sum(weights.get(f, 0.0) * s for f, s in fam.items())
        raw *= 1.0 + agreement_bonus * agree

        contra_fam = _family_scores(contradict_items, time_bucket)
        contra_raw = sum(weights.get(f, 0.0) * s for f, s in contra_fam.items())
        net = raw - contradiction_penalty * contra_raw
        prob = max(0.0, min(1.0, net / active_sum))
        if temperature > 0:
            prob = prob ** (1.0 / temperature)
        if raw <= 0:
            state = "negative" if contra_raw > 0 else "unknown"
        elif contra_raw > 0:
            state = "mixed"
        else:
            state = "positive"
        scores.append(
            {
                "candidate": brand,
                "score": round(net, 4),
                "prob": round(prob, 4),
                "support_score": round(raw, 4),
                "contradiction_score": round(contra_raw, 4),
                "net_evidence": round(net, 4),
                "evidence_state": state,
                "families": {f: round(s, 4) for f, s in fam.items()},
                "temporal_agreement": round(agree, 4),
                "supporting": [
                    {"family": it["family"],
                     "strength": round(float(it.get("strength", 0.0)), 4),
                     "quality": round(float(it["quality"]), 4) if it.get("quality") is not None else None,
                     "timestamp": it.get("timestamp"),
                     "frame_index": it.get("frame_index"),
                     "track_id": it.get("track_id"),
                     "persistence": it.get("persistence")}
                    for it in sorted(support_items, key=lambda e: -float(e.get("strength", 0.0)))[:8]
                ],
                "contradicting": [
                    {"family": it["family"],
                     "strength": round(float(it.get("strength", 0.0)), 4),
                     "quality": round(float(it["quality"]), 4) if it.get("quality") is not None else None,
                     "timestamp": it.get("timestamp"),
                     "frame_index": it.get("frame_index")}
                    for it in sorted(contradict_items, key=lambda e: -float(e.get("strength", 0.0)))[:8]
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
            "negative_evidence": False,
            "evidence_state": "unknown",
            "reason": "no candidate evidence across any modality (absence, not negative)",
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
        "negative_evidence": any(s["contradiction_score"] > 0 for s in scores),
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

    # Negative evidence: an explicit contradiction on the SAME candidate lowers
    # its net evidence and probability; absence is never penalised.
    contra = [
        *solo,
        {"family": "product", "strength": 0.8, "quality": 1.0,
         "timestamp": 0.0, "polarity": CONTRADICT},
    ]
    net = fuse_candidates({"N": contra})["ranking"][0]
    assert net["contradiction_score"] > 0, net
    assert net["net_evidence"] < net["support_score"], net
    assert net["evidence_state"] == "mixed", net
    assert abs(net["net_evidence"] - (net["support_score"]
           - _CONTRADICTION_PENALTY * net["contradiction_score"])) < 1e-3

    # Absence (no items at all) is "unknown", not "negative".
    empty = fuse_candidates({"N": []})
    assert empty["verdict"] == "abstain" and empty["evidence_state"] == "unknown"

    # Track collapse: repeated observations of ONE physical track count once.
    track = [{"family": "logo", "strength": 0.9, "quality": 1.0,
              "timestamp": t, "frame_index": t, "track_id": "t1"}
             for t in (0, 3, 6, 9)]
    assert fuse_candidates({"N": track})["ranking"][0]["families"]["logo"] > 0.9
    one = fuse_candidates({"N": [track[0]]})["ranking"][0]["families"]["logo"]
    assert fuse_candidates({"N": track})["ranking"][0]["families"]["logo"] >= one
    print("evidence_fusion self-check OK")


if __name__ == "__main__":
    _check()