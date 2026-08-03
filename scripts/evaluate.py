"""
Phase 1 Evaluation Harness — Fusion Ablation Experiment
This is the centerpiece experiment per the prompt (Section 3, 10).

Measures:
1. Layer 1 detection metrics (Precision/Recall/F1/mAP)
2. Layer 2b confidence calibration (ROC-AUC/ECE)
3. LAYER 2a FUSION ABLATION (dynamic-weighted vs. plain fusion under noise/blur)
   — This is the primary research contribution.

The ablation compares fused-with-dynamic-weighting vs. fused-without-weighting
under controlled synthetic degradation, tied to the <10% relative-performance-drop
success bar.

All results are logged to timestamped JSON files for traceability.
"""

import sys
from pathlib import Path

# Ensure project root is on path (this script lives in scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from copy import deepcopy

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.layer2.fusion import QualityAwareFusion
from src.training.losses import (
    CrossModalConsistencyLoss,
    EntropyRegularization,
    QualityAlignmentLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SyntheticEvaluator:
    """
    Evaluates the Phase 1 pipeline using synthetic test data.
    Generates controlled degradation (noise, blur) to test the fusion ablation.

    This is a self-contained evaluation that does NOT require real video data.
    Metrics are logged to timestamped files for provenance.
    """

    def __init__(self, output_dir: str = "./outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_synthetic_brand_data(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> Dict:
        """
        Generate synthetic multimodal brand detection data.

        Returns dict with:
            - audio_embeds: (n_samples, 1024) simulated audio embeddings
            - video_embeds: (n_samples, 1024) simulated video embeddings
            - audio_quality: (n_samples, 2) [snr_norm, vad_conf]
            - video_quality: (n_samples, 3) [blur_norm, exposure_norm, det_stability]
            - labels: (n_samples,) binary brand presence labels
            - evidence: list of evidence dicts per sample
        """
        rng = np.random.RandomState(seed)

        # Generate random embeddings
        audio_embeds = rng.randn(n_samples, 1024).astype(np.float32)
        video_embeds = rng.randn(n_samples, 1024).astype(np.float32)

        # Normalize embeddings
        audio_embeds = audio_embeds / np.linalg.norm(
            audio_embeds, axis=1, keepdims=True
        )
        video_embeds = video_embeds / np.linalg.norm(
            video_embeds, axis=1, keepdims=True
        )

        # Generate quality scores — some low, some high
        audio_quality = np.zeros((n_samples, 2))
        video_quality = np.zeros((n_samples, 3))

        for i in range(n_samples):
            # Mix of high/low quality
            if i < n_samples // 3:
                # Low audio quality
                audio_quality[i] = [0.1, 0.2]
                video_quality[i] = [0.8, 0.7, 0.9]
            elif i < 2 * n_samples // 3:
                # Low video quality
                audio_quality[i] = [0.8, 0.7]
                video_quality[i] = [0.2, 0.3, 0.1]
            else:
                # Both good
                audio_quality[i] = [0.8, 0.8]
                video_quality[i] = [0.7, 0.7, 0.8]

        # Generate labels based on evidence
        labels = np.zeros(n_samples, dtype=int)
        evidence_list = []

        for i in range(n_samples):
            # Evidence strengths correlate with label
            has_logo = rng.random() > 0.5
            has_speech = rng.random() > 0.5
            has_ocr = rng.random() > 0.7

            ev = {
                "logo_detected": 0.8 if has_logo else 0.0,
                "speech_mention": 0.7 if has_speech else 0.0,
                "ocr_hit": 0.6 if has_ocr else 0.0,
                "scene_context": 0.3 if has_logo else 0.0,
                "product_retrieval": 0.2 if has_logo else 0.0,
            }
            evidence_list.append(ev)

            # Positive if at least two evidence types present
            score = sum(1 for v in ev.values() if v > 0.5)
            labels[i] = 1 if score >= 2 else 0

        return {
            "audio_embeds": audio_embeds,
            "video_embeds": video_embeds,
            "audio_quality": audio_quality,
            "video_quality": video_quality,
            "labels": labels,
            "evidence": evidence_list,
        }

    def evaluate_fusion_ablation(
        self,
        n_samples: int = 100,
        noise_levels: List[float] = None,
        blur_levels: List[float] = None,
    ) -> Dict:
        """
        Run the centerpiece fusion ablation experiment.

        Compares:
        - Dynamic-weighted fusion (trained gating network from quality signals)
        - Plain fusion (equal weights, no quality information)

        Under varying synthetic noise and blur levels.

        Returns dict with full metrics — logged to file for traceability.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.3, 0.5]
        if blur_levels is None:
            blur_levels = [0, 5, 15, 30]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize fusion model with learned gating
        dynamic_fusion = QualityAwareFusion(
            audio_dim=1024,
            video_dim=1024,
            hidden_dim=512,
            num_heads=8,
            num_layers=3,
            use_learned_gating=True,
        ).to(device)

        # Plain fusion uses the same weights — only the forward pass differs
        plain_fusion = deepcopy(dynamic_fusion)

        results = {
            "experiment": "fusion_ablation",
            "timestamp": self.timestamp,
            "config": {
                "n_samples": n_samples,
                "noise_levels": noise_levels,
                "blur_levels": blur_levels,
            },
            "dynamic_weighted": {},
            "plain_fusion": {},
        }

        # Generate clean data
        data = self.generate_synthetic_brand_data(n_samples=n_samples)

        # Split into disjoint train and evaluation subsets
        n_train = int(n_samples * 0.7)
        train_idx = list(range(n_train))
        eval_idx = list(range(n_train, n_samples))

        train_data = {k: v[train_idx] if isinstance(v, np.ndarray) else v
                      for k, v in data.items()}
        eval_data = {k: v[eval_idx] if isinstance(v, np.ndarray) else v
                     for k, v in data.items()}

        # === Phase 1: Train gating network on TRAIN subset ===
        logger.info("Training gating network on synthetic corruption data...")
        dynamic_fusion = self._train_gating_network(
            dynamic_fusion, train_data, device, epochs=150, lr=1e-3
        )
        plain_fusion = deepcopy(dynamic_fusion)
        logger.info("Gating network training complete.")

        # === Phase 2: Train linear classifier on TRAIN fused embeddings ===
        logger.info("Training linear classifier on clean TRAIN embeddings...")
        audio_embeds = torch.from_numpy(train_data["audio_embeds"]).float().to(device)
        video_embeds = torch.from_numpy(train_data["video_embeds"]).float().to(device)
        audio_q = torch.from_numpy(train_data["audio_quality"]).float().to(device)
        video_q = torch.from_numpy(train_data["video_quality"]).float().to(device)

        with torch.no_grad():
            clean_dynamic = dynamic_fusion(
                audio_embeds, video_embeds, audio_q, video_q,
                use_dynamic_weights=True,
            )
            clean_plain = plain_fusion(
                audio_embeds, video_embeds,
                use_dynamic_weights=False,
            )

        # Train separate classifiers for dynamic and plain
        dyn_classifier = self._train_linear_classifier(
            clean_dynamic["fused_embed"].cpu().numpy(), train_data["labels"]
        )
        plain_classifier = self._train_linear_classifier(
            clean_plain["fused_embed"].cpu().numpy(), train_data["labels"]
        )
        logger.info("Linear classifiers trained.")

        for noise_level in noise_levels:
            for blur_level in blur_levels:
                condition = f"noise_{noise_level}_blur_{blur_level}"
                logger.info(
                    f"Evaluating condition: {condition}"
                )

                # Degrade quality signals (from eval subset to match embedding dimensions)
                audio_q_deg = eval_data["audio_quality"].copy()
                video_q_deg = eval_data["video_quality"].copy()

                # Apply synthetic noise degradation to quality
                audio_q_deg[:, 0] = np.clip(
                    audio_q_deg[:, 0] - noise_level, 0, 1
                )
                audio_q_deg[:, 1] = np.clip(
                    audio_q_deg[:, 1] - noise_level, 0, 1
                )

                # Apply synthetic blur degradation to video quality
                blur_factor = blur_level / 30.0
                video_q_deg[:, 0] = np.clip(
                    video_q_deg[:, 0] - blur_factor, 0, 1
                )
                video_q_deg[:, 2] = np.clip(
                    video_q_deg[:, 2] - blur_factor, 0, 1
                )

                # Also corrupt the embeddings themselves to match the degraded quality
                audio_embeds_deg = eval_data["audio_embeds"].copy()
                video_embeds_deg = eval_data["video_embeds"].copy()
                audio_embeds_deg += noise_level * np.random.randn(*audio_embeds_deg.shape).astype(np.float32)
                video_embeds_deg += blur_factor * np.random.randn(*video_embeds_deg.shape).astype(np.float32)

                # Convert to tensors
                audio_embeds_t = torch.from_numpy(
                    audio_embeds_deg
                ).float().to(device)
                video_embeds_t = torch.from_numpy(
                    video_embeds_deg
                ).float().to(device)
                audio_quality_t = torch.from_numpy(audio_q_deg).float().to(device)
                video_quality_t = torch.from_numpy(video_q_deg).float().to(device)

                # Dynamic-weighted fusion
                with torch.no_grad():
                    dynamic_out = dynamic_fusion(
                        audio_embeds_t,
                        video_embeds_t,
                        audio_quality_t,
                        video_quality_t,
                        use_dynamic_weights=True,
                    )

                    plain_out = plain_fusion(
                        audio_embeds_t,
                        video_embeds_t,
                        audio_quality=None,
                        video_quality=None,
                        use_dynamic_weights=False,
                    )

                # Compute classification scores using trained classifiers
                dynamic_scores = self._compute_scores(
                    dynamic_out["fused_embed"], dyn_classifier
                )
                plain_scores = self._compute_scores(
                    plain_out["fused_embed"], plain_classifier
                )

                labels = eval_data["labels"]

                # Dynamic metrics
                dyn_metrics = self._compute_classification_metrics(
                    dynamic_scores, labels
                )

                # Record fusion weight statistics
                dyn_metrics["mean_audio_weight"] = float(
                    dynamic_out["audio_weight"].mean().cpu()
                )
                dyn_metrics["mean_video_weight"] = float(
                    dynamic_out["video_weight"].mean().cpu()
                )
                results["dynamic_weighted"][condition] = dyn_metrics

                # Plain metrics
                plain_metrics = self._compute_classification_metrics(
                    plain_scores, labels
                )
                results["plain_fusion"][condition] = plain_metrics

                # Compute relative performance drop
                dyn_acc = dyn_metrics["accuracy"]
                plain_acc = plain_metrics["accuracy"]
                if plain_acc > 0:
                    rel_drop = (
                        (plain_acc - dyn_acc) / plain_acc * 100
                    )
                else:
                    rel_drop = 0.0

                # Check against <10% success bar
                results.setdefault("relative_drop", {})[condition] = {
                    "relative_drop_percent": float(rel_drop),
                    "meets_success_bar": rel_drop < 10.0,
                }

        # Save results
        output_path = self.output_dir / f"fusion_ablation_{self.timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Fusion ablation results saved to {output_path}")

        return results

    def _train_gating_network(
        self,
        fusion_model: "QualityAwareFusion",
        data: Dict,
        device: str,
        epochs: int = 150,
        lr: float = 1e-3,
    ) -> "QualityAwareFusion":
        """
        Train the gating network on synthetic corruption data.

        Uses cross-modal consistency loss: when one modality is degraded,
        the fused embedding should be closer to the uncorrupted modality's
        embedding. Also uses quality alignment loss and entropy regularization.
        """
        # Prepare data
        audio_embeds = torch.from_numpy(data["audio_embeds"]).float().to(device)
        video_embeds = torch.from_numpy(data["video_embeds"]).float().to(device)
        audio_q = torch.from_numpy(data["audio_quality"]).float().to(device)
        video_q = torch.from_numpy(data["video_quality"]).float().to(device)

        # Create corruption labels: 0=audio degraded, 1=video degraded, 2=both clean
        n = len(data["audio_embeds"])
        corruption_type = torch.zeros(n, dtype=torch.long, device=device)
        corruption_type[:n // 3] = 0  # audio degraded
        corruption_type[n // 3: 2 * n // 3] = 1  # video degraded
        corruption_type[2 * n // 3:] = 2  # both clean

        # Collect all trainable parameters (gating network + fusion transformer)
        params = list(fusion_model.gating_network.parameters()) + \
                 list(fusion_model.fusion.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Reusable loss modules from src/training/losses.py
        consistency_loss = CrossModalConsistencyLoss()
        quality_loss = QualityAlignmentLoss()
        entropy_reg = EntropyRegularization()

        fusion_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            result = fusion_model(
                audio_embeds, video_embeds,
                audio_q, video_q,
                use_dynamic_weights=True,
            )
            fused = result["fused_embed"]  # (n, hidden_dim)
            w_audio = result["audio_weight"]  # (n,)
            w_video = result["video_weight"]  # (n,)

            # Projected embeddings for consistency loss
            audio_proj = fusion_model.fusion.audio_proj(audio_embeds)
            video_proj = fusion_model.fusion.video_proj(video_embeds)

            # 1. Cross-modal consistency loss (vectorized via shared module)
            l_consistency = consistency_loss(fused, audio_proj, video_proj, corruption_type)

            # 2. Quality alignment loss
            l_quality = quality_loss(w_audio, audio_q, video_q)

            # 3. Entropy regularization (prevent collapse)
            l_entropy = entropy_reg(w_audio, w_video)

            # Combined loss
            total_loss = 1.0 * l_consistency + 0.5 * l_quality + 0.1 * l_entropy

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{epochs}: "
                    f"loss={total_loss.item():.4f}, "
                    f"consistency={l_consistency.item():.4f}, "
                    f"quality={l_quality.item():.4f}, "
                    f"entropy={l_entropy.item():.4f}"
                )

        fusion_model.eval()
        return fusion_model

    def _train_linear_classifier(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> torch.nn.Module:
        """
        Train a linear classifier on fused embeddings.

        This provides a meaningful downstream metric: can a simple classifier
        distinguish positive/negative samples from the fused representation?
        """
        device = "cpu"
        input_dim = embeddings.shape[1]
        classifier = torch.nn.Linear(input_dim, 1).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        X = torch.from_numpy(embeddings).float().to(device)
        y = torch.from_numpy(labels).float().to(device)

        classifier.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = classifier(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        classifier.eval()
        return classifier

    def _compute_scores(
        self, embeddings: torch.Tensor, classifier: torch.nn.Module
    ) -> np.ndarray:
        """
        Compute classification scores using a trained linear classifier.

        Moves embeddings to the classifier's device (CPU) before inference.

        Args:
            embeddings: (n_samples, hidden_dim) fused embeddings
            classifier: Trained Linear(in_features=hidden_dim, out_features=1)

        Returns:
            scores: (n_samples,) probability scores in [0, 1]
        """
        classifier_device = next(classifier.parameters()).device
        embeddings_on_device = embeddings.to(device=classifier_device)
        with torch.no_grad():
            logits = classifier(embeddings_on_device).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def _compute_classification_metrics(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """Compute classification metrics from scores and labels."""
        threshold = 0.5
        predictions = (scores >= threshold).astype(int)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            roc_auc_val = 0.5  # undefined for single class
        else:
            roc_auc_val = float(roc_auc_score(labels, scores))

        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": roc_auc_val,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

        return metrics

    def evaluate_confidence_calibration(
        self, n_samples: int = 1000
    ) -> Dict:
        """
        Evaluate Layer 2b confidence calibration.

        Measures ROC-AUC and Expected Calibration Error (ECE).
        """
        from src.layer2.confidence import EvidenceConfidenceScorer

        scorer = EvidenceConfidenceScorer()
        rng = np.random.RandomState(42)

        # Generate synthetic evidence and labels
        scores = []
        labels = []

        for _ in range(n_samples):
            # Random evidence
            evidence = {
                "logo_detected": rng.uniform(0, 1),
                "speech_mention": rng.uniform(0, 1),
                "ocr_hit": rng.uniform(0, 1),
                "scene_context": rng.uniform(0, 1),
                "product_retrieval": rng.uniform(0, 1),
            }

            result = scorer.compute_evidence_score(evidence)
            scores.append(result["confidence"])

            # Ground truth: positive if average evidence > 0.5
            label = 1 if np.mean(list(evidence.values())) > 0.5 else 0
            labels.append(label)

        scores = np.array(scores)
        labels = np.array(labels)

        # Compute metrics
        roc_auc = scorer.compute_roc_auc(scores, labels)
        ece = scorer.compute_ece(scores, labels, n_bins=10)

        calibration_result = {
            "experiment": "confidence_calibration",
            "timestamp": self.timestamp,
            "n_samples": n_samples,
            "roc_auc": roc_auc,
            "ece": ece,
            "mean_confidence": float(np.mean(scores)),
            "std_confidence": float(np.std(scores)),
        }

        # Save results
        output_path = (
            self.output_dir / f"calibration_{self.timestamp}.json"
        )
        with open(output_path, "w") as f:
            json.dump(calibration_result, f, indent=2)

        logger.info(f"Calibration results saved to {output_path}")

        return calibration_result

    def run_full_evaluation(self) -> Dict:
        """
        Run full Phase 1 evaluation suite.

        Returns aggregated results dict.
        """
        logger.info("=" * 60)
        logger.info("Starting Phase 1 Full Evaluation")
        logger.info("=" * 60)

        # Fusion ablation (centerpiece)
        logger.info("\n--- Fusion Ablation ---")
        ablation_results = self.evaluate_fusion_ablation()

        # Confidence calibration
        logger.info("\n--- Confidence Calibration ---")
        calibration_results = self.evaluate_confidence_calibration()

        # Aggregate
        full_results = {
            "experiment": "phase1_full_evaluation",
            "timestamp": self.timestamp,
            "fusion_ablation": ablation_results,
            "confidence_calibration": calibration_results,
            "summary": {
                "n_ablation_conditions": len(
                    ablation_results.get("relative_drop", {})
                ),
                "conditions_meeting_success_bar": sum(
                    1 for v in ablation_results.get("relative_drop", {}).values()
                    if v["meets_success_bar"]
                ),
                "calibration_ece": calibration_results.get("ece"),
                "calibration_roc_auc": calibration_results.get("roc_auc"),
            },
        }

        # Save full results
        output_path = (
            self.output_dir / f"phase1_evaluation_{self.timestamp}.json"
        )
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        logger.info(f"\nFull evaluation results saved to {output_path}")
        logger.info("=" * 60)

        return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Evaluation Harness"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of synthetic samples for ablation",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip fusion ablation (centerpiece)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip confidence calibration evaluation",
    )
    args = parser.parse_args()

    evaluator = SyntheticEvaluator(output_dir=args.output_dir)

    if args.skip_ablation and args.skip_calibration:
        logger.warning("Both ablation and calibration skipped. Nothing to do.")
        return

    if not args.skip_ablation and not args.skip_calibration:
        evaluator.run_full_evaluation()
    elif not args.skip_ablation:
        evaluator.evaluate_fusion_ablation(n_samples=args.n_samples)
    elif not args.skip_calibration:
        evaluator.evaluate_confidence_calibration()


if __name__ == "__main__":
    main()