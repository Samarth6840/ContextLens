"""
Layer 2b — Evidence-Based Confidence Scoring
Decomposed, explainable confidence score from multiple evidence types.

Design:
- Evidence sources: logo detection, speech mention, OCR hit, scene context, product retrieval
- Each evidence type has a configurable weight
- Per-modality quality weights from Layer 2a modulate evidence contributions
- Below minimum-evidence threshold, output "no confident evidence"
- Calibration via reliability diagrams / expected calibration error (ECE)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class EvidenceConfidenceScorer:
    """
    Computes decomposed, explainable confidence scores from multimodal evidence.

    Evidence types and their base weights (configurable):
        - logo_detected: 0.45
        - speech_mention: 0.20
        - ocr_hit: 0.15
        - scene_context: 0.10
        - product_retrieval: 0.10
    """

    def __init__(
        self,
        evidence_weights: Optional[Dict[str, float]] = None,
        min_evidence_threshold: float = 0.30,
        aggregation: str = "weighted_sum",
    ):
        self.evidence_weights = evidence_weights or {
            "logo_detected": 0.45,
            "speech_mention": 0.20,
            "ocr_hit": 0.15,
            "scene_context": 0.10,
            "product_retrieval": 0.10,
        }
        self.min_evidence_threshold = min_evidence_threshold
        self.aggregation = aggregation

        # Calibration model (fitted post-training)
        self.calibrator = None

    def compute_evidence_score(
        self,
        evidence: Dict[str, float],
        modality_quality_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Compute confidence score from evidence dict.

        Args:
            evidence: Dict mapping evidence type to strength (0-1).
                      Keys: logo_detected, speech_mention, ocr_hit,
                            scene_context, product_retrieval
            modality_quality_weights: Optional dict with 'audio_weight' and
                                      'video_weight' from Layer 2a fusion.
                                      Used to modulate evidence contributions.

        Returns:
            Dict with keys:
                - confidence: float (0-1) final confidence score
                - evidence_breakdown: dict of per-evidence contributions
                - is_confident: bool (True if above threshold)
                - status: str ("confident" or "no_confident_evidence")
        """
        # Modulate evidence by modality quality if available
        modulated_evidence = {}
        for ev_type, strength in evidence.items():
            weight = self.evidence_weights.get(ev_type, 0.0)

            # Apply modality quality modulation
            if modality_quality_weights is not None:
                if ev_type in ("logo_detected", "ocr_hit", "scene_context"):
                    # Video-dependent evidence
                    video_weight = modality_quality_weights.get(
                        "video_weight", 0.5
                    )
                    weight = weight * (0.5 + 0.5 * video_weight)
                elif ev_type == "speech_mention":
                    # Audio-dependent evidence
                    audio_weight = modality_quality_weights.get(
                        "audio_weight", 0.5
                    )
                    weight = weight * (0.5 + 0.5 * audio_weight)

            modulated_evidence[ev_type] = {
                "strength": float(strength),
                "base_weight": float(self.evidence_weights[ev_type]),
                "modulated_weight": float(weight),
                "contribution": float(strength * weight),
            }

        # Aggregate
        if self.aggregation == "weighted_sum":
            total_weight = sum(
                e["modulated_weight"] for e in modulated_evidence.values()
            )
            if total_weight > 0:
                confidence = sum(
                    e["contribution"] for e in modulated_evidence.values()
                ) / total_weight
            else:
                confidence = 0.0

        elif self.aggregation == "noisy_or":
            # Noisy-OR: 1 - prod(1 - p_i)
            confidence = 1.0
            for e in modulated_evidence.values():
                confidence *= 1.0 - (e["strength"] * e["modulated_weight"])
            confidence = 1.0 - confidence

        elif self.aggregation == "learned_calibration":
            # Weighted sum followed by calibration
            total_weight = sum(
                e["modulated_weight"] for e in modulated_evidence.values()
            )
            if total_weight > 0:
                raw_confidence = sum(
                    e["contribution"] for e in modulated_evidence.values()
                ) / total_weight
            else:
                raw_confidence = 0.0

            # Apply learned calibration if available
            if self.calibrator is not None:
                confidence = float(
                    self.calibrator.predict([[raw_confidence]])[0]
                )
            else:
                confidence = raw_confidence
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Threshold check
        is_confident = confidence >= self.min_evidence_threshold
        status = "confident" if is_confident else "no_confident_evidence"

        return {
            "confidence": confidence,
            "evidence_breakdown": modulated_evidence,
            "is_confident": is_confident,
            "status": status,
        }

    def fit_calibration(
        self,
        raw_scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict:
        """
        Fit calibration model using isotonic regression.

        Args:
            raw_scores: Array of uncalibrated confidence scores
            ground_truth: Array of binary ground truth labels

        Returns:
            Dict with calibration metrics
        """
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_scores, ground_truth)

        # Compute calibration metrics
        prob_true, prob_pred = calibration_curve(
            ground_truth, raw_scores, n_bins=10
        )

        # Expected Calibration Error
        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            "ece": float(ece),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

    def compute_ece(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            scores: Predicted confidence scores
            labels: Binary ground truth labels
            n_bins: Number of bins for calibration

        Returns:
            ECE value (lower is better)
        """
        prob_true, prob_pred = calibration_curve(
            labels, scores, n_bins=n_bins
        )
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        return ece

    def compute_roc_auc(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute ROC-AUC for confidence scoring.

        Args:
            scores: Predicted confidence scores
            labels: Binary ground truth labels

        Returns:
            ROC-AUC score
        """
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))