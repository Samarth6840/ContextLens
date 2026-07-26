"""
Layer 2a — Modality Quality Estimator
Estimates per-modality quality signals used for dynamic weighting in fusion.

Audio quality signals: SNR (dB), VAD confidence
Video quality signals: blur (Laplacian variance), exposure, detection-confidence variance

All estimates are computed from actual signal properties — no hardcoded values.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AudioQualityEstimator:
    """
    Estimates audio quality from waveform properties.
    Provides concrete signals: SNR and VAD confidence.
    """

    @staticmethod
    def estimate_snr(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Estimate Signal-to-Noise Ratio in dB.

        Uses a noise floor estimate from the lowest-energy percentile.
        """
        if audio is None or len(audio) == 0:
            return 0.0

        signal_power = np.mean(audio**2)
        if signal_power < 1e-10:
            return 0.0

        # Segment-based noise floor estimation
        segment_length = int(0.025 * sample_rate)  # 25ms
        num_segments = max(1, len(audio) // segment_length)
        segment_energies = np.array([
            np.mean(audio[i * segment_length : (i + 1) * segment_length] ** 2)
            for i in range(num_segments)
        ])

        noise_floor = float(np.percentile(segment_energies, 10))
        snr_db = 10.0 * np.log10(
            (signal_power + 1e-10) / (noise_floor + 1e-10)
        )

        return snr_db

    @staticmethod
    def estimate_vad_confidence(audio: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Estimate Voice Activity Detection confidence.

        Computes the fraction of active frames based on energy thresholding.
        Higher values indicate more speech content relative to silence.
        """
        if audio is None or len(audio) == 0:
            return 0.0

        segment_length = int(0.025 * sample_rate)  # 25ms
        num_segments = max(1, len(audio) // segment_length)
        segment_energies = np.array([
            np.mean(audio[i * segment_length : (i + 1) * segment_length] ** 2)
            for i in range(num_segments)
        ])

        if len(segment_energies) == 0:
            return 0.0

        noise_floor = float(np.percentile(segment_energies, 10))
        threshold = noise_floor * 2.0
        active_frames = float(np.sum(segment_energies > threshold))
        vad_confidence = min(1.0, active_frames / max(1, num_segments))

        return vad_confidence

    def estimate(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """
        Full audio quality estimation.

        Returns:
            Dict with keys: snr_db, vad_confidence, quality_score (0-1 normalized)
        """
        snr = self.estimate_snr(audio, sample_rate)
        vad = self.estimate_vad_confidence(audio, sample_rate)

        # Normalize SNR to [0, 1] using a sigmoid-like mapping
        # SNR of 0dB -> ~0.12, SNR of 15dB -> ~0.73, SNR of 30dB -> ~0.95
        snr_normalized = 1.0 / (1.0 + np.exp(-0.2 * (snr - 15.0)))

        # Combine signals into a single quality score
        quality_score = float(0.5 * snr_normalized + 0.5 * vad)

        return {
            "snr_db": float(snr),
            "vad_confidence": float(vad),
            "quality_score": quality_score,
        }


class VideoQualityEstimator:
    """
    Estimates video quality from frame properties.
    Provides concrete signals: blur, exposure, detection-confidence variance.
    """

    @staticmethod
    def estimate_blur(frame: np.ndarray) -> float:
        """
        Estimate blur using Laplacian variance.

        Lower variance = more blur.
        Threshold: < 100 is considered blurry.
        """
        if frame is None or frame.size == 0:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def estimate_exposure(frame: np.ndarray) -> Dict[str, float]:
        """
        Estimate exposure from mean pixel value.

        Returns dict with:
            - mean_pixel: mean pixel intensity (0-255)
            - is_under_exposed: bool flag (mean < 40)
            - is_over_exposed: bool flag (mean > 210)
        """
        if frame is None or frame.size == 0:
            return {"mean_pixel": 0.0, "is_under_exposed": True, "is_over_exposed": False}

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_pixel = float(np.mean(gray))

        return {
            "mean_pixel": mean_pixel,
            "is_under_exposed": mean_pixel < 40.0,
            "is_over_exposed": mean_pixel > 210.0,
        }

    def estimate(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Full video quality estimation for a single frame.

        Returns:
            Dict with keys: blur_score, mean_pixel, quality_score (0-1 normalized)
        """
        blur = self.estimate_blur(frame)
        exposure = self.estimate_exposure(frame)

        # Normalize blur: higher = better, cap at 500
        blur_normalized = min(1.0, blur / 500.0)

        # Normalize exposure: penalize under/over exposure
        mean_px = exposure["mean_pixel"]
        exposure_score = 1.0 - abs(mean_px - 127.0) / 127.0
        exposure_score = max(0.0, min(1.0, exposure_score))

        # Combine signals
        quality_score = float(0.5 * blur_normalized + 0.5 * exposure_score)

        return {
            "blur_score": blur,
            "mean_pixel": mean_px,
            "quality_score": quality_score,
        }

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """Estimate video quality for a batch of frames."""
        return [self.estimate(frame) for frame in frames]

    def estimate_aggregate(
        self,
        frames: List[np.ndarray],
        detection_confidence_variance: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Aggregate video quality across multiple frames.

        Args:
            frames: List of video frames
            detection_confidence_variance: Optional variance from detector stability

        Returns:
            Aggregated quality dict
        """
        if not frames:
            return {"quality_score": 0.0}

        per_frame = self.estimate_batch(frames)
        mean_blur = float(np.mean([q["blur_score"] for q in per_frame]))
        mean_exposure = float(np.mean([q["mean_pixel"] for q in per_frame]))

        # Normalize and combine
        blur_norm = min(1.0, mean_blur / 500.0)
        exposure_norm = 1.0 - abs(mean_exposure - 127.0) / 127.0
        exposure_norm = max(0.0, min(1.0, exposure_norm))

        # Detection stability: low variance = stable = high quality
        det_stability = 1.0
        if detection_confidence_variance is not None:
            det_stability = 1.0 - min(1.0, detection_confidence_variance)

        quality_score = float(
            0.3 * blur_norm + 0.3 * exposure_norm + 0.4 * det_stability
        )

        return {
            "mean_blur_score": mean_blur,
            "mean_pixel": mean_exposure,
            "detection_stability": det_stability,
            "quality_score": quality_score,
        }