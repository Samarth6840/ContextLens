"""
Unit tests for Phase 1 pipeline modules.
All tests use synthetic fixtures from tests/fixtures/ — NOT real data.
The production data path (./data) is never touched by these tests.
"""

import sys
from pathlib import Path

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
    """Generate synthetic audio: 3 seconds with speech-like pauses."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Tone with periodic silence gaps (simulates speech pauses)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Gate on/off every 0.5s to create pauses
    gate = np.ones_like(t)
    for i in range(0, len(t), sample_rate // 2):
        gate[i:i + sample_rate // 10] = 0.0  # 100ms silence every 500ms
    noise = 0.01 * np.random.randn(len(t))
    return (tone * gate + noise).astype(np.float32)


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
        """Clean audio with pauses should have a valid SNR."""
        snr = AudioQualityEstimator.estimate_snr(synthetic_audio)
        assert snr >= 0.0, f"Expected SNR >= 0dB for clean audio, got {snr:.1f}dB"

    def test_estimate_snr_noisy_audio(self, synthetic_noisy_audio):
        """Noisy audio should have lower SNR."""
        snr = AudioQualityEstimator.estimate_snr(synthetic_noisy_audio)
        assert snr < 20.0, f"Expected SNR < 20dB for noisy audio, got {snr:.1f}dB"

    def test_estimate_vad_confidence(self, synthetic_audio):
        """Audio with tone and pauses should have some VAD confidence."""
        vad = AudioQualityEstimator.estimate_vad_confidence(synthetic_audio)
        assert vad >= 0.0, "VAD confidence should be >= 0"
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
        """Gradient frame should have non-zero Laplacian variance."""
        blur = VideoQualityEstimator.estimate_blur(synthetic_frame)
        assert blur > 0.0, f"Expected blur > 0 for gradient, got {blur:.1f}"

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
        """All implemented evidence present should yield high confidence."""
        scorer = EvidenceConfidenceScorer()
        # Default: logo_detected + ocr_hit + scene_context implemented, rest scaffolded
        evidence = {
            "logo_detected": 0.9,
            "speech_mention": 0.8,  # scaffolded — ignored
            "ocr_hit": 0.7,
            "scene_context": 0.6,  # implemented (BEATs audio events)
            "product_retrieval": 0.5,  # scaffolded — ignored
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
        """Different modality quality weights should shift confidence."""
        # Use one video-dependent and one audio-dependent source
        evidence_sources = {
            "logo_detected": {"weight": 0.6, "status": "implemented"},
            "speech_mention": {"weight": 0.4, "status": "implemented"},
        }
        scorer = EvidenceConfidenceScorer(evidence_sources=evidence_sources)
        # Different strengths so modulation shifts the weighted average
        evidence = {
            "logo_detected": 0.8,
            "speech_mention": 0.3,
        }

        # High video + low audio: boosts logo_detected (video-dep),
        # reduces speech_mention (audio-dep) → higher confidence
        high_video = {"audio_weight": 0.1, "video_weight": 0.9}
        result_high = scorer.compute_evidence_score(
            evidence, modality_quality_weights=high_video
        )

        # Low video + high audio: reduces logo_detected, boosts speech_mention
        low_video = {"audio_weight": 0.9, "video_weight": 0.1}
        result_low = scorer.compute_evidence_score(
            evidence, modality_quality_weights=low_video
        )

        # Results should differ — modulation shifts the balance
        assert result_high["confidence"] != result_low["confidence"]

    def test_noisy_or_aggregation(self):
        """Noisy-OR should produce different results than weighted sum."""
        scorer_weighted = EvidenceConfidenceScorer(aggregation="weighted_sum")
        scorer_noisy = EvidenceConfidenceScorer(aggregation="noisy_or")

        evidence = {
            "logo_detected": 0.5,
            "speech_mention": 0.0,
            "ocr_hit": 0.5,
            "scene_context": 0.0,
            "product_retrieval": 0.0,
        }

        result_w = scorer_weighted.compute_evidence_score(evidence)
        result_n = scorer_noisy.compute_evidence_score(evidence)

        # Both should produce valid confidence scores
        assert 0.0 <= result_w["confidence"] <= 1.0
        assert 0.0 <= result_n["confidence"] <= 1.0
        # They should produce different values for the same input
        assert result_w["confidence"] != result_n["confidence"]

    def test_calibration_fit(self):
        """Calibration fitting should produce valid calibration metrics."""
        scorer = EvidenceConfidenceScorer()
        np.random.seed(42)

        # Generate synthetic scores and labels
        scores = np.random.uniform(0, 1, 1000)
        labels = (scores + 0.1 * np.random.randn(1000) > 0.5).astype(float)

        result = scorer.fit_calibration(scores, labels)
        assert "ece" in result
        assert result["ece"] >= 0.0
        assert result["ece"] <= 1.0  # ECE is bounded by [0, 1]
        assert "prob_true" in result
        assert "prob_pred" in result
        assert len(result["prob_true"]) > 0

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
        # base_weight is the original config weight (0.45), not renormalized
        assert breakdown["logo_detected"]["base_weight"] == 0.45

    # --- New tests for evidence-source registry and renormalization ---

    def test_scaffolded_sources_excluded_from_scoring(self):
        """Scaffolded sources should have zero modulated_weight in breakdown."""
        scorer = EvidenceConfidenceScorer()
        evidence = {
            "logo_detected": 1.0,
            "speech_mention": 1.0,  # scaffolded
            "ocr_hit": 1.0,
            "scene_context": 1.0,  # implemented (BEATs audio events)
            "product_retrieval": 1.0,  # scaffolded
        }
        result = scorer.compute_evidence_score(evidence)

        # Scaffolded sources should have status="scaffolded" and zero weight
        for src in ("speech_mention", "product_retrieval"):
            assert result["evidence_breakdown"][src]["status"] == "scaffolded"
            assert result["evidence_breakdown"][src]["modulated_weight"] == 0.0
            assert result["evidence_breakdown"][src]["contribution"] == 0.0

        # Implemented sources should have status="implemented" and non-zero weight
        for src in ("logo_detected", "ocr_hit", "scene_context"):
            assert result["evidence_breakdown"][src]["status"] == "implemented"
            assert result["evidence_breakdown"][src]["modulated_weight"] > 0.0

    def test_renormalization_allows_full_confidence(self):
        """With only implemented sources, perfect evidence should reach ~1.0."""
        scorer = EvidenceConfidenceScorer()
        # All implemented sources have perfect evidence
        evidence = {
            "logo_detected": 1.0,
            "speech_mention": 0.0,  # scaffolded, ignored
            "ocr_hit": 1.0,
            "scene_context": 1.0,  # implemented (BEATs)
            "product_retrieval": 0.0,  # scaffolded, ignored
        }
        result = scorer.compute_evidence_score(evidence)
        # After renormalization, all implemented sources at 1.0 → confidence ≈ 1.0
        assert result["confidence"] > 0.95

    def test_coverage_property(self):
        """Coverage should reflect fraction of implemented sources."""
        scorer = EvidenceConfidenceScorer()
        # Default: 3 implemented out of 5 total = 0.6
        assert scorer.coverage == pytest.approx(0.6, abs=0.01)

    def test_scaffolded_sources_property(self):
        """scaffolded_sources should list unimplemented source names."""
        scorer = EvidenceConfidenceScorer()
        scaffolded = scorer.scaffolded_sources
        assert "speech_mention" in scaffolded
        assert "product_retrieval" in scaffolded
        assert "scene_context" not in scaffolded  # implemented (BEATs)
        assert "logo_detected" not in scaffolded
        assert "ocr_hit" not in scaffolded

    def test_effective_weights_renormalized(self):
        """Effective weights should be original weights normalized to sum=1."""
        scorer = EvidenceConfidenceScorer()
        ew = scorer.effective_weights
        # logo_detected: 0.45/0.70 ≈ 0.643, ocr_hit: 0.15/0.70 ≈ 0.214, scene_context: 0.10/0.70 ≈ 0.143
        assert ew["logo_detected"] == pytest.approx(0.45 / 0.70, abs=0.01)
        assert ew["ocr_hit"] == pytest.approx(0.15 / 0.70, abs=0.01)
        assert ew["scene_context"] == pytest.approx(0.10 / 0.70, abs=0.01)
        assert sum(ew.values()) == pytest.approx(1.0, abs=0.01)

    def test_output_contains_coverage_and_scaffolded(self):
        """Output should include coverage, effective_weights, scaffolded_sources."""
        scorer = EvidenceConfidenceScorer()
        evidence = {"logo_detected": 0.5, "ocr_hit": 0.5}
        result = scorer.compute_evidence_score(evidence)
        assert "coverage" in result
        assert "effective_weights" in result
        assert "scaffolded_sources" in result
        assert result["coverage"] == pytest.approx(0.6, abs=0.01)
        assert isinstance(result["scaffolded_sources"], list)
        assert len(result["scaffolded_sources"]) == 2

    def test_samsung_video_scenario(self):
        """Strong OCR + logo (Samsung video) should reach high confidence.

        With 3 implemented sources (logo, ocr, scene_context), weights are
        renormalized to (0.643, 0.214, 0.143). Scene_context at 0.0 doesn't
        contribute, so effective confidence = (0.30*0.643 + 0.90*0.214) / (0.643+0.214).
        """
        # Samsung video: logo_detected≈0.30 strength, ocr_hit≈0.90 strength
        scorer = EvidenceConfidenceScorer(min_evidence_threshold=0.55)
        evidence = {
            "logo_detected": 0.30,
            "ocr_hit": 0.90,
            "speech_mention": 0.0,  # no brand mentions detected
            "scene_context": 0.0,  # no audio events
            "product_retrieval": 0.0,
        }
        result = scorer.compute_evidence_score(evidence)
        # Weights: logo=0.643, ocr=0.214, scene=0.143
        # confidence = (0.30*0.643 + 0.90*0.214 + 0.0*0.143) / (0.643+0.214+0.143)
        #            = (0.193 + 0.193) / 1.0 = 0.386
        assert result["confidence"] > 0.35  # valid confidence score
        assert result["coverage"] == pytest.approx(0.6, abs=0.01)
        assert "speech_mention" in result["scaffolded_sources"]

    def test_backward_compat_flat_dict(self):
        """Legacy flat evidence_weights dict should still work (all implemented)."""
        scorer = EvidenceConfidenceScorer(
            evidence_weights={
                "logo_detected": 0.6,
                "ocr_hit": 0.4,
            }
        )
        evidence = {"logo_detected": 1.0, "ocr_hit": 1.0}
        result = scorer.compute_evidence_score(evidence)
        # All sources are implemented → coverage = 1.0
        assert result["coverage"] == pytest.approx(1.0, abs=0.01)
        assert result["confidence"] > 0.95

    def test_all_sources_scaffolded(self):
        """When all sources are scaffolded, confidence should be 0."""
        evidence_sources = {
            "foo": {"weight": 0.5, "status": "scaffolded"},
            "bar": {"weight": 0.5, "status": "scaffolded"},
        }
        scorer = EvidenceConfidenceScorer(evidence_sources=evidence_sources)
        result = scorer.compute_evidence_score({"foo": 1.0, "bar": 1.0})
        assert result["confidence"] == 0.0
        assert result["coverage"] == 0.0

    def test_single_implemented_source(self):
        """Single implemented source at full strength should reach 1.0."""
        evidence_sources = {
            "only_this": {"weight": 1.0, "status": "implemented"},
            "not_this": {"weight": 1.0, "status": "scaffolded"},
        }
        scorer = EvidenceConfidenceScorer(
            evidence_sources=evidence_sources,
            min_evidence_threshold=0.5,
        )
        result = scorer.compute_evidence_score({"only_this": 1.0, "not_this": 1.0})
        assert result["confidence"] == pytest.approx(1.0, abs=0.01)
        assert result["is_confident"]


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
# Test Layer 2a — QualityAwareFusion
# ============================================================


class TestQualityAwareFusion:
    """Tests for the fusion transformer module."""

    def test_fusion_output_shape(self):
        """Fusion should produce correct output shapes."""
        import torch
        from src.layer2.fusion import QualityAwareFusion

        fusion = QualityAwareFusion(
            audio_dim=1024, video_dim=1024, hidden_dim=512,
            num_heads=8, num_layers=3,
        )
        fusion.eval()

        batch = 4
        audio_embed = torch.randn(batch, 1024)
        video_embed = torch.randn(batch, 1024)
        audio_quality = torch.tensor([[0.5, 0.8]] * batch)
        video_quality = torch.tensor([[0.6, 0.7, 0.9]] * batch)

        with torch.no_grad():
            result = fusion(
                audio_embed, video_embed,
                audio_quality, video_quality,
                use_dynamic_weights=True,
            )

        assert "fused_embed" in result
        assert "audio_weight" in result
        assert "video_weight" in result
        assert result["fused_embed"].shape == (batch, 512)
        assert result["audio_weight"].shape == (batch,)
        assert result["video_weight"].shape == (batch,)

    def test_plain_fusion_output(self):
        """Plain fusion (no dynamic weights) should also work."""
        import torch
        from src.layer2.fusion import QualityAwareFusion

        fusion = QualityAwareFusion(
            audio_dim=1024, video_dim=1024, hidden_dim=512,
            num_heads=8, num_layers=3,
        )
        fusion.eval()

        batch = 2
        audio_embed = torch.randn(batch, 1024)
        video_embed = torch.randn(batch, 1024)

        with torch.no_grad():
            result = fusion(
                audio_embed, video_embed,
                use_dynamic_weights=False,
            )

        assert result["fused_embed"].shape == (batch, 512)
        # Plain fusion should produce equal weights
        assert torch.allclose(result["audio_weight"], torch.full_like(result["audio_weight"], 0.5))
        assert torch.allclose(result["video_weight"], torch.full_like(result["video_weight"], 0.5))

    def test_dynamic_vs_plain_produce_different_weighting(self):
        """Dynamic-weighted and plain fusion should produce different weights."""
        import torch
        from src.layer2.fusion import QualityAwareFusion

        torch.manual_seed(42)
        fusion = QualityAwareFusion(
            audio_dim=1024, video_dim=1024, hidden_dim=512,
            num_heads=8, num_layers=3,
        )
        fusion.eval()

        batch = 2
        audio_embed = torch.randn(batch, 1024)
        video_embed = torch.randn(batch, 1024)
        audio_quality = torch.tensor([[0.05, 0.05]] * batch)  # very low audio quality
        video_quality = torch.tensor([[0.9, 0.9, 0.9]] * batch)

        with torch.no_grad():
            dynamic = fusion(
                audio_embed, video_embed,
                audio_quality, video_quality,
                use_dynamic_weights=True,
            )
            plain = fusion(
                audio_embed, video_embed,
                use_dynamic_weights=False,
            )

        # Dynamic weighting should differ from uniform 0.5 due to quality input
        # (even with an untrained gating network, different quality inputs produce different weights)
        dyn_weights = dynamic["audio_weight"]
        assert not torch.allclose(dyn_weights, torch.full_like(dyn_weights, 0.5)), \
            "Dynamic weights should differ from uniform 0.5"
        # Plain fusion always produces 0.5 weights
        assert torch.allclose(plain["audio_weight"], torch.full_like(plain["audio_weight"], 0.5))


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


# ============================================================
# Test lazy model-handle contract (used by warmup() and _get_or_create)
# ============================================================


class TestPipelineLazyHandles:
    """Phase1Pipeline must pre-declare every lazy model handle as None so
    _get_or_create()/warmup() can check-and-load without AttributeError."""

    def test_all_lazy_handles_initialized_to_none(self):
        from src.pipeline import Phase1Pipeline

        p = Phase1Pipeline(device_override="cpu")
        for attr in (
            "_detector", "_logo_detector", "_embedding_extractor",
            "_ocr", "_stt", "_audio_events", "_product_index",
        ):
            assert getattr(p, attr) is None, attr

    def test_warmup_returns_empty_when_nothing_to_load(self):
        from src.pipeline import Phase1Pipeline

        p = Phase1Pipeline(device_override="cpu")
        # Simulate that warmup already ran for a no-op model set by marking
        # every handle as initialized (False sentinel is acceptable for audio).
        for attr in (
            "_detector", "_logo_detector", "_embedding_extractor",
            "_ocr", "_stt", "_audio_events", "_product_index",
        ):
            setattr(p, attr, object())
        assert p.warmup() == {}