"""
Layer 2b — Evidence-Based Confidence Scoring
Decomposed, explainable confidence score from multiple evidence types.

Design:
- Evidence sources: logo detection, speech mention, OCR hit, scene context, product retrieval
- Each evidence type has a configurable weight and implementation status
- Non-implemented (scaffolded) sources are excluded; implemented-source weights
  are renormalized so strong evidence from available sources can reach [0, 1]
- Per-modality quality weights from Layer 2a modulate evidence contributions
- Below minimum-evidence threshold, output "no confident evidence"
- Calibration via reliability diagrams / expected calibration error (ECE)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Status constants
STATUS_IMPLEMENTED = "implemented"
STATUS_SCAFFOLDED = "scaffolded"


class EvidenceConfidenceScorer:
    """
    Computes decomposed, explainable confidence scores from multimodal evidence.

    Evidence sources and their base weights (configurable via evidence_sources):
        - logo_detected: 0.45 (implemented)
        - speech_mention: 0.20 (scaffolded — Layer 2c not built)
        - ocr_hit: 0.15 (implemented)
        - scene_context: 0.10 (implemented — BEATs audio event detector)
        - product_retrieval: 0.10 (scaffolded — product retrieval not built)

    Scaffolded sources are excluded from scoring. Implemented-source weights
    are renormalized to sum to 1.0 so that perfect evidence from available
    sources can reach a confidence of 1.0.
    """

    def __init__(
        self,
        evidence_weights: Optional[Dict[str, float]] = None,
        min_evidence_threshold: float = 0.30,
        aggregation: str = "weighted_sum",
        evidence_sources: Optional[Dict[str, dict]] = None,
    ):
        # Parse evidence_sources config (preferred) or fall back to evidence_weights
        if evidence_sources is not None:
            self._source_registry = {}
            for name, spec in evidence_sources.items():
                if isinstance(spec, dict):
                    raw_status = spec.get("status", "")
                    if raw_status == STATUS_IMPLEMENTED:
                        status = STATUS_IMPLEMENTED
                    else:
                        # Treat missing, invalid, or unknown statuses as scaffolded
                        # so they cannot silently become implemented without explicit
                        # configuration review.
                        status = STATUS_SCAFFOLDED
                    self._source_registry[name] = {
                        "weight": float(spec.get("weight", 0.0)),
                        "status": status,
                    }
                else:
                    # Flat dict of weights — treat all as implemented
                    self._source_registry[name] = {
                        "weight": float(spec),
                        "status": STATUS_IMPLEMENTED,
                    }
        elif evidence_weights is not None:
            # Legacy flat dict — all sources assumed implemented
            self._source_registry = {
                name: {"weight": float(w), "status": STATUS_IMPLEMENTED}
                for name, w in evidence_weights.items()
            }
        else:
            # Defaults
            self._source_registry = {
                "logo_detected": {"weight": 0.45, "status": STATUS_IMPLEMENTED},
                "speech_mention": {"weight": 0.20, "status": STATUS_SCAFFOLDED},
                "ocr_hit": {"weight": 0.15, "status": STATUS_IMPLEMENTED},
                "scene_context": {"weight": 0.10, "status": STATUS_IMPLEMENTED},
                "product_retrieval": {"weight": 0.10, "status": STATUS_SCAFFOLDED},
            }

        self.min_evidence_threshold = min_evidence_threshold
        self.aggregation = aggregation

        # Split into implemented / scaffolded
        self._implemented = {
            name: info
            for name, info in self._source_registry.items()
            if info["status"] == STATUS_IMPLEMENTED
        }
        self._scaffolded = {
            name: info
            for name, info in self._source_registry.items()
            if info["status"] == STATUS_SCAFFOLDED
        }

        # Renormalize implemented-source weights to sum to 1.0.
        # Scaffolded sources get an explicit 0.0 in the output mapping so
        # callers see every registry key and know the system is aware of them.
        impl_normalized: Dict[str, float] = {}
        total_impl_weight = sum(info["weight"] for info in self._implemented.values())
        if total_impl_weight > 0:
            impl_normalized = {
                name: info["weight"] / total_impl_weight
                for name, info in self._implemented.items()
            }
        else:
            impl_normalized = {name: 0.0 for name in self._implemented}

        self._effective_weights: Dict[str, float] = {}
        for name in self._source_registry:
            if name in self._implemented:
                self._effective_weights[name] = impl_normalized[name]
            else:
                self._effective_weights[name] = 0.0

        self._coverage = (
            len(self._implemented) / len(self._source_registry)
            if self._source_registry
            else 0.0
        )

        logger.info(
            "EvidenceConfidenceScorer: %d/%d sources implemented (%.0f%% coverage). "
            "Effective weights: %s",
            len(self._implemented),
            len(self._source_registry),
            self._coverage * 100,
            {k: round(v, 4) for k, v in self._effective_weights.items()},
        )
        if self._scaffolded:
            logger.info(
                "Scaffolded (zero-contribution) sources: %s",
                list(self._scaffolded.keys()),
            )

        # Calibration model (fitted post-training)
        self.calibrator = None

    @property
    def effective_weights(self) -> Dict[str, float]:
        """Effective weights for every registry source: normalized for implemented,
        zero for scaffolded. Every config key is present."""
        return dict(self._effective_weights)

    @property
    def coverage(self) -> float:
        """Fraction of evidence sources that are implemented (0.0–1.0)."""
        return self._coverage

    @property
    def scaffolded_sources(self) -> List[str]:
        """Names of evidence sources that are scaffolded but not built."""
        return list(self._scaffolded.keys())

    def compute_evidence_score(
        self,
        evidence: Dict[str, float],
        modality_quality_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Compute confidence score from evidence dict.

        Only implemented sources contribute to the score. Scaffolded sources
        are included in the breakdown with status="scaffolded" and zero weight.

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
                - coverage: float (fraction of sources implemented)
                - effective_weights: dict of renormalized weights used
                - scaffolded_sources: list of source names not yet built
        """
        # Build breakdown for ALL sources (implemented + scaffolded)
        modulated_evidence = {}

        # Process only implemented sources for scoring
        for ev_type, strength in evidence.items():
            if ev_type not in self._source_registry:
                continue

            source_info = self._source_registry[ev_type]
            is_scaffolded = source_info["status"] == STATUS_SCAFFOLDED

            if is_scaffolded:
                # Scaffolded: record in breakdown but with zero weight
                modulated_evidence[ev_type] = {
                    "strength": float(strength),
                    "base_weight": float(source_info["weight"]),
                    "modulated_weight": 0.0,
                    "contribution": 0.0,
                    "status": STATUS_SCAFFOLDED,
                }
                continue

            # Implemented source — use renormalized weight
            weight = self._effective_weights.get(ev_type, 0.0)

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
                elif ev_type == "product_retrieval":
                    # Product retrieval depends on both modalities
                    audio_weight = modality_quality_weights.get("audio_weight", 0.5)
                    video_weight = modality_quality_weights.get("video_weight", 0.5)
                    combined = 0.5 * audio_weight + 0.5 * video_weight
                    weight = weight * (0.5 + 0.5 * combined)

            modulated_evidence[ev_type] = {
                "strength": float(strength),
                "base_weight": float(source_info["weight"]),
                "modulated_weight": float(weight),
                "contribution": float(strength * weight),
                "status": STATUS_IMPLEMENTED,
            }

        # Aggregate using only implemented sources
        impl_entries = {
            k: v for k, v in modulated_evidence.items()
            if v.get("status") == STATUS_IMPLEMENTED
        }

        if self.aggregation == "weighted_sum":
            total_weight = sum(
                e["modulated_weight"] for e in impl_entries.values()
            )
            if total_weight > 0:
                confidence = sum(
                    e["contribution"] for e in impl_entries.values()
                ) / total_weight
            else:
                confidence = 0.0

        elif self.aggregation == "noisy_or":
            total_weight = sum(
                e["modulated_weight"] for e in impl_entries.values()
            )
            confidence = 1.0
            for e in impl_entries.values():
                confidence *= 1.0 - (e["strength"] * e["modulated_weight"])
            confidence = 1.0 - confidence

        elif self.aggregation == "learned_calibration":
            total_weight = sum(
                e["modulated_weight"] for e in impl_entries.values()
            )
            if total_weight > 0:
                raw_confidence = sum(
                    e["contribution"] for e in impl_entries.values()
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
            "total_modulated_weight": total_weight,
            "is_confident": is_confident,
            "status": status,
            "coverage": self._coverage,
            "effective_weights": dict(self._effective_weights),
            "scaffolded_sources": self.scaffolded_sources,
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

        # Compute calibration metrics — guard against edge cases
        unique_labels = np.unique(ground_truth)
        if len(unique_labels) < 2:
            return {
                "ece": 0.0,
                "prob_true": [],
                "prob_pred": [],
            }

        try:
            prob_true, prob_pred = calibration_curve(
                ground_truth, raw_scores, n_bins=10
            )
            if len(prob_true) == 0:
                return {"ece": 0.0, "prob_true": [], "prob_pred": []}
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
        except ValueError:
            return {"ece": 0.0, "prob_true": [], "prob_pred": []}

        return {
            "ece": ece,
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
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        try:
            prob_true, prob_pred = calibration_curve(
                labels, scores, n_bins=n_bins
            )
            if len(prob_true) == 0:
                return 0.0
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
        except ValueError:
            return 0.0

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
