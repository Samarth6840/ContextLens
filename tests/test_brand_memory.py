import math
import os
import tempfile

import pytest

from src.layer2.brand_memory import BrandMemoryBank, load_or_new


def test_record_and_query():
    bank = BrandMemoryBank()
    bank.record("Apple", video_id="v1", frame=10, confidence=0.9, modality="visual")
    bank.record("apple", video_id="v1", frame=20, confidence=0.95, modality="visual")
    assert bank.size() == 1  # case-insensitive canonicalization
    ent = bank.get("APPLE")
    assert ent["sightings"] == 2
    assert ent["max_confidence"] == pytest.approx(0.95)
    assert ent["modalities"]["visual"] == 2


def test_recency_wins_without_embedding():
    bank = BrandMemoryBank(recency_decay=0.5)
    bank.record("NIKE", confidence=0.9, wall_time=10)
    bank.record("ADIDAS", confidence=0.6, wall_time=1)
    resolved = bank.resolve_reference("this pair of shoes", ref_clock=10.5)
    assert resolved is not None
    assert resolved["brand"] == "NIKE"  # more recent beats higher? no: recency-decay
    # At clock=10.5 both are recent; NIKE has higher confidence and same decayed age -> wins.


def test_older_high_confidence_beats_weak_recent_without_decay():
    bank = BrandMemoryBank(recency_decay=1.0)  # no decay -> pure max confidence
    bank.record("NIKE", confidence=0.95, wall_time=0)
    bank.record("ADIDAS", confidence=0.5, wall_time=100)
    resolved = bank.resolve_reference("this pair of shoes", ref_clock=101)
    assert resolved["brand"] == "NIKE"


def test_decay_demotes_old_memory():
    bank = BrandMemoryBank(recency_decay=0.5)
    bank.record("NIKE", confidence=0.99, wall_time=0)
    bank.record("ADIDAS", confidence=0.5, wall_time=1000)
    resolved = bank.resolve_reference("this phone here", ref_clock=1010)
    # Old Nike decays to ~0 huge, ADIDAS recent wins.
    assert resolved["brand"] == "ADIDAS"


def test_embedding_similarity_biases_selection():
    bank = BrandMemoryBank(recency_decay=1.0)
    bank.record("APPLE", confidence=0.5, embedding=[0.1, 0.1], wall_time=0)
    bank.record("SONY", confidence=0.8, embedding=[-0.9, 0.9], wall_time=0)
    # Reference embedding close to APPLE should override the higher-confidence SONY
    # when the embedding weight is dominant.
    resolved = bank.resolve_reference(
        "this phone", embedding=[0.12, 0.12], ref_clock=1, sim_weight=0.9
    )
    assert resolved["brand"] == "APPLE"


def test_reference_excludes_returned_products():
    bank = BrandMemoryBank(recency_decay=1.0)
    bank.record("APPLE", confidence=0.9, wall_time=0)
    bank.record("SONY", confidence=0.7, wall_time=0)
    resolved = bank.resolve_reference(
        "this phone", ref_clock=1, returned_products=["APPLE"]
    )
    assert resolved["brand"] == "SONY"


def test_no_resolution_when_no_candidates():
    bank = BrandMemoryBank()
    assert bank.resolve_reference("this phone") is None


def test_serialization_roundtrip_and_load_or_new():
    bank = BrandMemoryBank(recency_decay=0.25)
    bank.record("NIKE", video_id="v1", confidence=0.8, embedding=[1.0, 2.0, 3.0])
    bank.record("PUMA", video_id="v1", confidence=0.7)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "memory.json")
    bank.save(path)
    loaded = BrandMemoryBank.load(path)
    assert loaded.size() == 2
    assert loaded.recency_decay == 0.25
    assert loaded.get("NIKE")["embedding"] == [1.0, 2.0, 3.0]
    assert loaded.resolve_reference("these") and loaded.resolve_reference("these")["brand"]

    # load_or_new: existing path loads; missing path returns a fresh empty bank.
    reloaded = load_or_new(path)
    assert reloaded.size() == 2
    fresh = load_or_new(os.path.join(tmpdir, "does_not_exist.json"))
    assert fresh.size() == 0


def test_most_recent_and_best_brand():
    bank = BrandMemoryBank(recency_decay=0.5)
    bank.record("NIKE", confidence=0.6, wall_time=1)
    bank.record("ADIDAS", confidence=0.4, wall_time=5)
    assert bank.most_recent_brand() == "ADIDAS"
    # best = highest recency-decayed score at the latest clock.
    assert bank.best_brand() == "ADIDAS"
