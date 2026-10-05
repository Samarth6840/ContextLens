from src.layer2.evidence_fusion import (
    CONTRADICT,
    derive_cross_modal_contradictions,
    fuse_candidates,
)

SOLO = [{"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.0}]


def test_dependency_discount_same_bucket():
    dup = [
        {"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.0},
        {"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.4},
    ]
    a = fuse_candidates({"N": SOLO})["ranking"][0]["families"]["logo"]
    b = fuse_candidates({"N": dup})["ranking"][0]["families"]["logo"]
    assert a == b


def test_temporal_agreement_bonus():
    two = [*SOLO, {"family": "ocr", "strength": 0.9, "quality": 1.0, "timestamp": 0.5}]
    assert fuse_candidates({"N": two})["ranking"][0]["prob"] > fuse_candidates({"N": SOLO})["ranking"][0]["prob"]


def test_ambiguous_on_close_margin():
    rival = [{"family": "logo", "strength": 0.88, "quality": 1.0, "timestamp": 0.0}]
    assert fuse_candidates({"N": SOLO, "A": rival})["verdict"] == "ambiguous"


def test_abstain_on_empty():
    assert fuse_candidates({})["verdict"] == "abstain"


def test_low_quality_reduces_score():
    weak = [{"family": "logo", "strength": 0.9, "quality": 0.15, "timestamp": 0.0}]
    assert fuse_candidates({"N": weak})["ranking"][0]["families"]["logo"] < 0.9


def test_contradiction_erodes_net_evidence():
    contra = [
        *SOLO,
        {"family": "product", "strength": 0.8, "quality": 1.0,
         "timestamp": 0.0, "polarity": CONTRADICT},
    ]
    pos = fuse_candidates({"N": SOLO})["ranking"][0]
    net = fuse_candidates({"N": contra})["ranking"][0]
    assert net["contradiction_score"] > 0
    assert net["net_evidence"] < pos["net_evidence"]
    assert net["prob"] < pos["prob"]
    assert net["evidence_state"] == "mixed"
    assert fuse_candidates({"N": contra})["negative_evidence"] is True


def test_absence_is_unknown_not_negative():
    # No candidate at all -> abstain, and it is NOT reported as negative
    # evidence (absence must never become a silent penalty).
    res = fuse_candidates({"N": []})
    assert res["verdict"] == "abstain"
    assert res["evidence_state"] == "unknown"
    assert res["negative_evidence"] is False


def _logo(s=0.9, q=0.9, t=0.0):
    return {"family": "logo", "strength": s, "quality": q, "timestamp": t}


def test_cross_modal_product_vs_logo_contradicts_wordmark():
    ledger = {
        "NIKE": [_logo(0.9)],
        "ADIDAS": [{"family": "product", "strength": 0.9, "quality": 1.0,
                    "timestamp": 0.0}],
    }
    out = derive_cross_modal_contradictions(ledger)
    assert any(it.get("polarity") == CONTRADICT for it in out["NIKE"])
    assert any(
        it.get("contradiction_rule") == "product_vs_logo" for it in out["NIKE"]
    )
    # input untouched
    assert all(it.get("polarity") != CONTRADICT for it in ledger["NIKE"])
    # and it actually lowers Nike's net evidence
    def _nike(rows):
        return next(r for r in rows["ranking"] if r["candidate"] == "NIKE")

    plain = _nike(fuse_candidates(ledger))
    net = _nike(fuse_candidates(out))
    assert net["net_evidence"] < plain["net_evidence"]
    assert net["contradiction_score"] > 0


def test_cross_modal_asr_vs_visual_contradicts():
    ledger = {
        "NIKE": [_logo(0.9)],
        "ADIDAS": [{"family": "speech", "strength": 1.0, "quality": 1.0,
                    "timestamp": None}],
    }
    out = derive_cross_modal_contradictions(ledger)
    assert any(
        it.get("contradiction_rule") == "asr_vs_visual" for it in out["NIKE"]
    )


def test_cross_modal_visual_vs_asr_contradicts_audio_only():
    ledger = {
        "NIKE": [{"family": "speech", "strength": 1.0, "quality": 1.0,
                  "timestamp": None}],
        "ADIDAS": [_logo(0.9)],
    }
    out = derive_cross_modal_contradictions(ledger)
    assert any(
        it.get("contradiction_rule") == "visual_vs_asr" for it in out["NIKE"]
    )


def test_cross_modal_no_contradiction_when_rival_is_consistent():
    # A rival seen in BOTH audio and video is consistent, not contradictory.
    ledger = {
        "NIKE": [_logo(0.9)],
        "ADIDAS": [_logo(0.85), {"family": "speech", "strength": 1.0,
                                 "quality": 1.0, "timestamp": None}],
    }
    out = derive_cross_modal_contradictions(ledger)
    assert not any(it.get("polarity") == CONTRADICT for it in out["NIKE"])


def test_cross_modal_respects_product_similarity_threshold():
    ledger = {
        "NIKE": [_logo(0.9)],
        "ADIDAS": [{"family": "product", "strength": 0.4, "quality": 1.0,
                    "timestamp": 0.0}],
    }
    out = derive_cross_modal_contradictions(ledger, min_product_similarity=0.6)
    assert not any(it.get("polarity") == CONTRADICT for it in out["NIKE"])


def test_track_collapse_credits_persistence():
    one = fuse_candidates({
        "N": [{"family": "logo", "strength": 0.9, "quality": 1.0,
               "timestamp": 0.0, "frame_index": 0, "track_id": "t1"}]
    })["ranking"][0]["families"]["logo"]
    long_track = fuse_candidates({
        "N": [{"family": "logo", "strength": 0.9, "quality": 1.0,
               "timestamp": t, "frame_index": t, "track_id": "t1"}
              for t in (0, 3, 6, 9)]
    })["ranking"][0]
    # One physical track seen longer is stronger, but never counted as N votes.
    assert long_track["families"]["logo"] > one
    assert long_track["families"]["logo"] < 0.9 * (1 + 0.25 * 4)