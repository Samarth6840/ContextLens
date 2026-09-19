from src.layer2.evidence_fusion import fuse_candidates

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