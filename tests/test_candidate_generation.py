from src.layer2.candidate_generation import generate_candidates, select_candidates


def test_generates_from_multiple_sources():
    cands = generate_candidates(
        logo_brands=["NIKE"],
        ocr_texts=["SAMSUNG GALAXY"],
        speech_mentions=[{"brand": "ADIDAS"}],
        product_matches=[{"brand": "PUMA", "similarity": 0.9}],
    )
    names = {c["candidate"] for c in cands}
    assert {"NIKE", "SAMSUNG", "ADIDAS", "PUMA"} <= names
    # scores are descending
    scores = [c["score"] for c in cands]
    assert scores == sorted(scores, reverse=True)
    # a direct logo resolution outranks a text-only alias
    nike = next(c for c in cands if c["candidate"] == "NIKE")
    samsung = next(c for c in cands if c["candidate"] == "SAMSUNG")
    assert nike["score"] > samsung["score"]


def test_unknown_text_is_not_generated():
    cands = generate_candidates(ocr_texts=["lorem ipsum", "random words"])
    assert cands == []


def test_top_k_truncates():
    cands = generate_candidates(
        logo_brands=["NIKE", "ADIDAS", "PUMA", "SAMSUNG"], top_k=2
    )
    assert len(cands) == 2


def test_select_candidates_protects_direct_evidence():
    ledger = {
        "NIKE": [{"family": "logo", "strength": 0.9}],
        "ADIDAS": [{"family": "ocr", "strength": 0.5}],
        "PUMA": [{"family": "ocr", "strength": 0.4}],
        "SAMSUNG": [{"family": "ocr", "strength": 0.3}],
    }
    generated = [
        {"candidate": "PUMA", "score": 0.6},
        {"candidate": "ADIDAS", "score": 0.5},
    ]
    # top_k=2 keeps the two generated, and NIKE because it has a logo (direct).
    kept = select_candidates(ledger, generated, top_k=2)
    assert set(kept) == {"NIKE", "PUMA", "ADIDAS"}
    assert "SAMSUNG" not in kept


def test_top_k_zero_disables_gating():
    ledger = {"A": [{"family": "ocr", "strength": 0.1}]}
    assert select_candidates(ledger, [], top_k=0) is ledger
