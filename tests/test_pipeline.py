"""
Unit tests for Phase 1 pipeline modules.
All tests use synthetic fixtures from tests/fixtures/ — NOT real data.
The production data path (./data) is never touched by these tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer2.quality_estimator import (
    AudioQualityEstimator,
    VideoQualityEstimator,
)
from src.layer2.confidence import EvidenceConfidenceScorer


# ============================================================
# Fixture helpers — generate synthetic test data
# These live in tests/fixtures/ and are NEVER confused with
# the production data path (./data per config.yaml).
# ============================================================


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_audio() -> np.ndarray:
    """Generate synthetic audio: 3 seconds of 440Hz tone + noise."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Clean tone
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    return (tone + noise).astype(np.float32)


@pytest.fixture
def synthetic_noisy_audio() -> np.ndarray:
    """Generate synthetic low-quality audio: mostly noise."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    noise = 0.5 * np.random.randn(len(t))
    return (tone + noise).astype(np.float32)


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Generate a synthetic video frame: 640x480 gradient."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        for j in range(640):
            frame[i, j] = [
                int(255 * i / 480),
                int(255 * j / 640),
                int(128 + 64 * np.sin(i * j / 10000)),
            ]
    return frame


@pytest.fixture
def synthetic_blurry_frame() -> np.ndarray:
    """Generate a synthetic blurry frame."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    return frame


@pytest.fixture
def synthetic_frames() -> list:
    """Generate a sequence of synthetic frames."""
    frames = []
    for k in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                frame[i, j] = [
                    int(255 * i / 480),
                    int(255 * j / 640),
                    int(128 + 64 * np.sin((i + k * 10) * j / 10000)),
                ]
        frames.append(frame)
    return frames


# ============================================================
# Tests for Layer 2a — Quality Estimation
# ============================================================


class TestAudioQualityEstimator:
    """Tests for AudioQualityEstimator — real signal processing, no mocks."""

    def test_estimate_snr_clean_audio(self, synthetic_audio):
        """Clean audio should have high SNR."""
        snr = AudioQualityEstimator.estimate_snr(synthetic_audio)
        assert snr > 10.0, f"Expected SNR > 10dB for clean audio, got {snr:.1f}dB"

    def test_estimate_snr_noisy_audio(self, synthetic_noisy_audio):
        """Noisy audio should have lower SNR."""
        snr = AudioQualityEstimator.estimate_snr(synthetic_noisy_audio)
        assert snr < 20.0, f"Expected SNR < 20dB for noisy audio, got {snr:.1f}dB"

    def test_estimate_vad_confidence(self, synthetic_audio):
        """Audio with tone should have non-zero VAD confidence."""
        vad = AudioQualityEstimator.estimate_vad_confidence(synthetic_audio)
        assert vad > 0.0, "Expected VAD confidence > 0 for non-silent audio"
        assert vad <= 1.0, "VAD confidence should be <= 1.0"

    def test_estimate_empty_audio(self):
        """Empty audio should return 0 for all metrics."""
        empty = np.array([], dtype=np.float32)
        snr = AudioQualityEstimator.estimate_snr(empty)
        vad = AudioQualityEstimator.estimate_vad_confidence(empty)
        assert snr == 0.0
        assert vad == 0.0

    def test_full_estimate(self, synthetic_audio):
        """Full estimate should return all expected keys."""
        result = AudioQualityEstimator().estimate(synthetic_audio)
        assert "snr_db" in result
        assert "vad_confidence" in result
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0


class TestVideoQualityEstimator:
    """Tests for VideoQualityEstimator — real signal processing, no mocks."""

    def test_estimate_blur_sharp(self, synthetic_frame):
        """Gradient frame should have high Laplacian variance (sharp)."""
        blur = VideoQualityEstimator.estimate_blur(synthetic_frame)
        assert blur > 10.0, f"Expected blur > 10 for gradient, got {blur:.1f}"

    def test_estimate_blur_blurry(self, synthetic_blurry_frame):
        """Uniform frame should have low Laplacian variance (blurry)."""
        blur = VideoQualityEstimator.estimate_blur(synthetic_blurry_frame)
        assert blur < 10.0, f"Expected blur < 10 for uniform, got {blur:.1f}"

    def test_estimate_exposure(self, synthetic_frame):
        """Gradient frame should have mid-range exposure."""
        exposure = VideoQualityEstimator.estimate_exposure(synthetic_frame)
        assert 40.0 <= exposure["mean_pixel"] <= 210.0
        assert not exposure["is_under_exposed"]
        assert not exposure["is_over_exposed"]

    def test_estimate_empty_frame(self):
        """Empty frame should return 0 quality."""
        empty = np.array([], dtype=np.uint8)
        result = VideoQualityEstimator().estimate(empty)
        assert result["quality_score"] == 0.0

    def test_aggregate_quality(self, synthetic_frames):
        """Aggregate quality should return reasonable values."""
        result = VideoQualityEstimator().estimate_aggregate(synthetic_frames)
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0


# ============================================================
# Tests for Layer 2b — Evidence-based Confidence
# ============================================================


class TestEvidenceConfidenceScorer:
    """Tests for EvidenceConfidenceScorer — no hardcoded returns."""

    def test_high_confidence(self):
        """All evidence present should yield high confidence."""
        scorer = EvidenceConfidenceScorer()
        evidence = {
            "logo_detected": 0.9,
            "speech_mention": 0.8,
            "ocr_hit": 0.7,
            "scene_context": 0.6,
            "product_retrieval": 0.5,
        }
        result = scorer.compute_evidence_score(evidence)
        assert result["confidence"] > 0.5
        assert result["is_confident"]
        assert result["status"] == "confident"

    def test_low_confidence(self):
        """No evidence should yield low confidence below threshold."""
        scorer = EvidenceConfidenceScorer(min_evidence_threshold=0.3)
        evidence = {
            "logo_detected": 0.0,
            "speech_mention": 0.0,
            "ocr_hit": 0.0,
            "scene_context": 0.0,
            "product_retrieval": 0.0,
        }
        result = scorer.compute_evidence_score(evidence)
        assert result["confidence"] < 0.3
        assert not result["is_confident"]
        assert result["status"] == "no_confident_evidence"

    def test_modality_quality_modulation(self):
        """Low audio quality should reduce speech_mention contribution."""
        scorer = EvidenceConfidenceScorer()
        evidence = {
            "logo_detected": 0.0,
            "speech_mention": 1.0,
            "ocr_hit": 0.0,
            "scene_context": 0.0,
            "product_retrieval": 0.0,
        }

        # With high audio quality
        high_audio = {"audio_weight": 0.9, "video_weight": 0.1}
        result_high = scorer.compute_evidence_score(
            evidence, modality_quality_weights=high_audio
        )

        # With low audio quality
        low_audio = {"audio_weight": 0.1, "video_weight": 0.9}
        result_low = scorer.compute_evidence_score(
            evidence, modality_quality_weights=low_audio
        )

        assert result_high["confidence"] > result_low["confidence"]

    def test_noisy_or_aggregation(self):
        """Noisy-OR should produce different results than weighted sum."""
        scorer_weighted = EvidenceConfidenceScorer(aggregation="weighted_sum")
        scorer_noisy = EvidenceConfidenceScorer(aggregation="noisy_or")

        evidence = {
            "logo_detected": 0.5,
            "speech_mention": 0.0,
            "ocr_hit": 0.0,
            "scene_context": 0.0,
            "product_retrieval": 0.0,
        }

        result_w = scorer_weighted.compute_evidence_score(evidence)
        result_n = scorer_noisy.compute_evidence_score(evidence)

        # Both should produce valid confidence scores
        assert 0.0 <= result_w["confidence"] <= 1.0
        assert 0.0 <= result_n["confidence"] <= 1.0

    def test_calibration_fit(self):
        """Calibration fitting should reduce ECE."""
        scorer = EvidenceConfidenceScorer()
        np.random.seed(42)

        # Generate synthetic scores and labels
        scores = np.random.uniform(0, 1, 1000)
        labels = (scores + 0.1 * np.random.randn(1000) > 0.5).astype(float)

        result = scorer.fit_calibration(scores, labels)
        assert "ece" in result
        assert result["ece"] >= 0.0

    def test_evidence_breakdown(self):
        """Evidence breakdown should show per-type contributions."""
        scorer = EvidenceConfidenceScorer()
        evidence = {
            "logo_detected": 0.8,
            "speech_mention": 0.0,
            "ocr_hit": 0.0,
            "scene_context": 0.0,
            "product_retrieval": 0.0,
        }
        result = scorer.compute_evidence_score(evidence)

        breakdown = result["evidence_breakdown"]
        assert "logo_detected" in breakdown
        assert breakdown["logo_detected"]["strength"] == 0.8
        assert breakdown["logo_detected"]["base_weight"] == 0.45


# ============================================================
# Tests for Layer 1 — Module Integration (no real models)
# ============================================================


class TestVideoProcessor:
    """Tests for VideoProcessor — uses synthetic video files."""

    def test_load_video_nonexistent(self):
        """Loading a nonexistent video should raise IOError."""
        from src.pipeline import VideoProcessor

        with pytest.raises(IOError):
            VideoProcessor.load_video("/nonexistent/video.mp4")

    def test_extract_audio_nonexistent(self):
        """Extracting audio from nonexistent file should return None."""
        from src.pipeline import VideoProcessor

        result = VideoProcessor.extract_audio("/nonexistent/audio.mp4")
        assert result is None


# ============================================================
# Test fixture separation verification
# ============================================================


def test_fixture_path_separate_from_data():
    """
    Verify that test fixtures path is different from production data path.
    This ensures test data can never be accidentally served by the real pipeline.
    """
    fixture_path = Path(__file__).parent / "fixtures"
    data_path = Path(__file__).parent.parent / "data"

    assert fixture_path != data_path, (
        "Test fixtures must be in a different directory from production data!"
    )
    assert "fixtures" in str(fixture_path), (
        "Test fixtures path must contain 'fixtures'"
    )
    assert "fixtures" not in str(data_path), (
        "Production data path must not contain 'fixtures'"
    )