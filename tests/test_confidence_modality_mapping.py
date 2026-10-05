"""Modality modulation must follow the modality the evidence actually came from.

`scene_context` and `audio_event` are both derived from BEATs audio events
(pipeline._aggregate_evidence builds scene_strength from `audio_events`). They
were modulated by video_weight, so a blurry or degraded VIDEO leg discounted a
perfectly clear AUDIO cue — the gate was modulating evidence by a modality that
had nothing to do with producing it.
"""

import yaml

from src.layer2.confidence import EvidenceConfidenceScorer

# Use the SHIPPED config, not the no-arg default registry. pipeline.confidence_
# scorer passes layer2b.evidence_sources; the class default is a stale 5-source
# fallback that would make this test assert against a config nothing ships with.
_CFG = yaml.safe_load(open("config/config.yaml"))["layer2b"]


def _contributions(scores: dict, quality: dict):
    scorer = EvidenceConfidenceScorer(
        evidence_sources=_CFG.get("evidence_sources"),
        min_evidence_threshold=_CFG["min_evidence_threshold"],
        aggregation=_CFG["aggregation"],
    )
    out = scorer.compute_evidence_score(scores, modality_quality_weights=quality)
    return {k: v["modulated_weight"] for k, v in out["evidence_breakdown"].items()}


def test_audio_evidence_ignores_degraded_video():
    """Video weight 0.0 must not change audio-sourced evidence at all."""
    good_video = {"video_weight": 1.0, "audio_weight": 1.0}
    bad_video = {"video_weight": 0.0, "audio_weight": 1.0}

    for ev_type in ("scene_context", "audio_event", "speech_mention"):
        a = _contributions({ev_type: 1.0}, good_video)[ev_type]
        b = _contributions({ev_type: 1.0}, bad_video)[ev_type]
        assert a == b, (
            f"{ev_type} is built from BEATs audio but its weight changed with "
            f"video_weight ({a} -> {b}); it is being modulated by a modality "
            "that did not produce it"
        )


def test_visual_evidence_ignores_degraded_audio():
    good_audio = {"video_weight": 1.0, "audio_weight": 1.0}
    bad_audio = {"video_weight": 1.0, "audio_weight": 0.0}

    for ev_type in ("logo_detected", "ocr_hit"):
        a = _contributions({ev_type: 1.0}, good_audio)[ev_type]
        b = _contributions({ev_type: 1.0}, bad_audio)[ev_type]
        assert a == b, (
            f"{ev_type} is frame-derived but its weight changed with "
            f"audio_weight ({a} -> {b})"
        )


def test_each_evidence_type_still_reacts_to_its_own_modality():
    """Guard against 'fixing' it by dropping modulation altogether."""
    audio_src = _contributions({"scene_context": 1.0}, {"video_weight": 1.0, "audio_weight": 1.0})
    audio_bad = _contributions({"scene_context": 1.0}, {"video_weight": 1.0, "audio_weight": 0.0})
    assert audio_src["scene_context"] > audio_bad["scene_context"], (
        "scene_context no longer responds to audio quality at all"
    )

    video_src = _contributions({"logo_detected": 1.0}, {"video_weight": 1.0, "audio_weight": 1.0})
    video_bad = _contributions({"logo_detected": 1.0}, {"video_weight": 0.0, "audio_weight": 1.0})
    assert video_src["logo_detected"] > video_bad["logo_detected"], (
        "logo_detected no longer responds to video quality at all"
    )
