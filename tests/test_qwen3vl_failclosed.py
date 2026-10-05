"""Qwen3-VL must degrade, never raise, when its weights are absent.

The 32B weights are intentionally not shipped (project policy forbids
auto-downloading ~60GB). Everything downstream already handles model=None, so a
missing checkpoint must never turn into a failed upload — only a fallback dict.
This is the safety guarantee; pin it so a future refactor cannot turn the
missing-weights path into an exception.
"""
import numpy as np
import pytest

from src.layer1.qwen3vl import Qwen3VL32B


def _absent_model(name="definitely-not-a-real-repo-xyz"):
    m = Qwen3VL32B(model_name=name, device="cpu")
    m._initialize()
    assert m.model is None
    return m


def test_missing_weights_do_not_raise_and_flag_fallback():
    m = _absent_model()
    out = m.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert out["fallback"] is True
    assert out["detections"] == []
    assert out["embeddings"].size == 0


def test_missing_weights_manufacturer_lookup_fails_closed():
    m = _absent_model()
    out = m.resolve_product_manufacturer(np.zeros((8, 8, 3), dtype=np.uint8))
    assert out["manufacturer"] is None
    assert out["fallback"] is True


def test_initialize_is_idempotent_after_fail_closed():
    m = _absent_model()
    m._initialize()  # must not raise or retry forever
    assert m._initialized is True
    assert m.analyze_frame(np.zeros((8, 8, 3), dtype=np.uint8))["fallback"] is True
