"""
Layer 2a — Modality-Quality-Aware Fusion Transformer
The primary research contribution of this project.

Design:
1. Per-modality quality estimation (from quality_estimator.py)
2. Learned gating network that produces modality weights from quality signals
3. Cross-attention transformer over weighted modality representations
4. Ablation: compare dynamic-weighted fusion vs. plain fusion (no weighting)

This is NOT a fixed heuristic — the gating network learns to weight modalities
based on actual estimated quality, enabling graceful degradation under noise.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LearnedGatingNetwork(nn.Module):
    """
    Learned gating network that produces modality weights from quality signals.

    Input: quality features for each modality (e.g., [snr, vad_conf] for audio,
           [blur_norm, exposure_norm, det_stability] for video)
    Output: softmax-normalized weights per modality

    This is trained end-to-end with the fusion transformer.
    """

    def __init__(
        self,
        audio_quality_dim: int = 2,
        video_quality_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.audio_quality_dim = audio_quality_dim
        self.video_quality_dim = video_quality_dim

        # Quality encoders
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_quality_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.video_encoder = nn.Sequential(
            nn.Linear(video_quality_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gating network — produces logits for softmax weighting
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 modalities: audio, video
        )

    def forward(
        self,
        audio_quality: torch.Tensor,
        video_quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_quality: (batch, audio_quality_dim) quality features
            video_quality: (batch, video_quality_dim) quality features

        Returns:
            weights: (batch, 2) softmax-normalized modality weights
        """
        audio_feat = self.audio_encoder(audio_quality)
        video_feat = self.video_encoder(video_quality)

        # Concatenate and produce gating logits
        combined = torch.cat([audio_feat, video_feat], dim=-1)
        gate_logits = self.gate(combined)

        # Softmax over modalities
        weights = F.softmax(gate_logits, dim=-1)
        return weights


class FixedHeuristicWeighting:
    """
    Fixed heuristic weighting baseline for ablation comparison.
    Uses simple rules instead of learned weights.
    """

    @staticmethod
    def compute_weights(
        audio_quality: Dict[str, float],
        video_quality: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Compute modality weights using fixed heuristics.

        If audio quality < 0.3, weight video more heavily.
        If video quality < 0.3, weight audio more heavily.
        Otherwise, equal weighting.
        """
        audio_q = audio_quality.get("quality_score", 0.5)
        video_q = video_quality.get("quality_score", 0.5)

        if audio_q < 0.3 and video_q >= 0.3:
            return (0.2, 0.8)  # audio poor, trust video
        elif video_q < 0.3 and audio_q >= 0.3:
            return (0.8, 0.2)  # video poor, trust audio
        else:
            return (0.5, 0.5)  # equal weighting


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention transformer that fuses weighted modality representations.

    Takes audio and video embeddings, weights them by modality quality,
    and applies cross-attention to produce a fused representation.
    """

    def __init__(
        self,
        audio_dim: int = 1024,  # Whisper encoder output dim
        video_dim: int = 1024,  # DINOv2 output dim
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project modalities to common hidden dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        # Cross-attention transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        audio_embed: torch.Tensor,
        video_embed: torch.Tensor,
        audio_weight: float = 0.5,
        video_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            audio_embed: (batch, audio_dim) audio embedding
            video_embed: (batch, video_dim) video embedding
            audio_weight: scalar weight for audio modality
            video_weight: scalar weight for video modality

        Returns:
            fused_embed: (batch, hidden_dim) fused representation
        """
        # Project to common dimension
        audio_h = self.audio_proj(audio_embed)  # (batch, hidden_dim)
        video_h = self.video_proj(video_embed)  # (batch, hidden_dim)

        # Apply modality weights
        audio_h = audio_h * audio_weight
        video_h = video_h * video_weight

        # Stack as sequence: [audio_token, video_token]
        # Shape: (batch, seq_len=2, hidden_dim)
        tokens = torch.stack([audio_h, video_h], dim=1)

        # Apply cross-attention transformer
        fused = self.transformer(tokens)  # (batch, 2, hidden_dim)

        # Aggregate: mean pool over sequence dimension
        fused = fused.mean(dim=1)  # (batch, hidden_dim)

        # Final projection
        fused = self.output_proj(fused)

        return fused


class QualityAwareFusion(nn.Module):
    """
    Complete quality-aware fusion module combining:
    1. Quality estimation signals
    2. Learned gating network for dynamic weighting
    3. Cross-attention fusion transformer

    This is the primary research contribution — the ablation compares
    dynamic-weighted fusion vs. plain fusion (equal weights).
    """

    def __init__(
        self,
        audio_dim: int = 1024,
        video_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_learned_gating: bool = True,
    ):
        super().__init__()
        self.use_learned_gating = use_learned_gating

        # Learned gating network
        self.gating_network = LearnedGatingNetwork(
            audio_quality_dim=2,  # snr, vad_confidence
            video_quality_dim=3,  # blur_norm, exposure_norm, det_stability
            hidden_dim=hidden_dim // 4,
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            audio_dim=audio_dim,
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Fixed heuristic baseline (for ablation)
        self.fixed_heuristic = FixedHeuristicWeighting()

    def forward(
        self,
        audio_embed: torch.Tensor,
        video_embed: torch.Tensor,
        audio_quality: Optional[torch.Tensor] = None,
        video_quality: Optional[torch.Tensor] = None,
        use_dynamic_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional dynamic weighting.

        Args:
            audio_embed: (batch, audio_dim) audio embedding
            video_embed: (batch, video_dim) video embedding
            audio_quality: (batch, 2) [snr_norm, vad_conf] or None
            video_quality: (batch, 3) [blur_norm, exposure_norm, det_stability] or None
            use_dynamic_weights: if True, use learned gating; if False, use equal weights

        Returns:
            Dict with keys:
                - fused_embed: (batch, hidden_dim) fused representation
                - audio_weight: (batch,) or scalar
                - video_weight: (batch,) or scalar
        """
        batch_size = audio_embed.shape[0]

        if use_dynamic_weights and audio_quality is not None and video_quality is not None:
            if self.use_learned_gating:
                # Learned gating
                weights = self.gating_network(audio_quality, video_quality)
                audio_weight = weights[:, 0]
                video_weight = weights[:, 1]
            else:
                # Fixed heuristic per item in batch
                audio_weights = []
                video_weights = []
                for i in range(batch_size):
                    aq = {
                        "snr_db": float(audio_quality[i, 0]),
                        "vad_confidence": float(audio_quality[i, 1]),
                        "quality_score": float(
                            0.5 * audio_quality[i, 0] + 0.5 * audio_quality[i, 1]
                        ),
                    }
                    vq = {
                        "blur_score": float(video_quality[i, 0]),
                        "mean_pixel": float(video_quality[i, 1]),
                        "quality_score": float(video_quality[i, 2]),
                    }
                    aw, vw = self.fixed_heuristic.compute_weights(aq, vq)
                    audio_weights.append(aw)
                    video_weights.append(vw)

                audio_weight = torch.tensor(
                    audio_weights, device=audio_embed.device
                )
                video_weight = torch.tensor(
                    video_weights, device=audio_embed.device
                )
        else:
            # Equal weighting (plain fusion baseline)
            audio_weight = torch.full(
                (batch_size,), 0.5, device=audio_embed.device
            )
            video_weight = torch.full(
                (batch_size,), 0.5, device=audio_embed.device
            )

        # Apply fusion with per-item weights
        fused_embeds = []
        for i in range(batch_size):
            fused = self.fusion(
                audio_embed[i : i + 1],
                video_embed[i : i + 1],
                audio_weight=float(audio_weight[i]),
                video_weight=float(video_weight[i]),
            )
            fused_embeds.append(fused)

        fused_embed = torch.cat(fused_embeds, dim=0)

        return {
            "fused_embed": fused_embed,
            "audio_weight": audio_weight,
            "video_weight": video_weight,
        }

    def forward_ablation(
        self,
        audio_embed: torch.Tensor,
        video_embed: torch.Tensor,
        audio_quality: torch.Tensor,
        video_quality: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run both dynamic-weighted and plain fusion for ablation comparison.

        Returns both results so the evaluation harness can compare them.
        """
        dynamic_result = self.forward(
            audio_embed=audio_embed,
            video_embed=video_embed,
            audio_quality=audio_quality,
            video_quality=video_quality,
            use_dynamic_weights=True,
        )

        plain_result = self.forward(
            audio_embed=audio_embed,
            video_embed=video_embed,
            audio_quality=None,
            video_quality=None,
            use_dynamic_weights=False,
        )

        return {
            "dynamic_fused": dynamic_result["fused_embed"],
            "dynamic_audio_weight": dynamic_result["audio_weight"],
            "dynamic_video_weight": dynamic_result["video_weight"],
            "plain_fused": plain_result["fused_embed"],
        }