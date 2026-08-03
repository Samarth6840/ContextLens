"""
Loss functions for training the quality-aware gating network.

Three losses combined:
1. Cross-modal consistency: fused embed should match the uncorrupted modality
2. Quality alignment: gating weights should track modality quality scores
3. Entropy regularization: prevent collapse to 0.5/0.5 or 1.0/0.0
"""

import torch
import torch.nn as nn


class CrossModalConsistencyLoss(nn.Module):
    """
    When one modality is corrupted, the fused embedding should be closer
    to the uncorrupted modality's projected embedding.

    Uses MSE between the fused output and the target (uncorrupted) modality.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        fused_embed: torch.Tensor,
        audio_proj: torch.Tensor,
        video_proj: torch.Tensor,
        corruption_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_embed: (n, hidden_dim) fused embeddings
            audio_proj: (n, hidden_dim) projected audio embeddings
            video_proj: (n, hidden_dim) projected video embeddings
            corruption_type: (n,) long tensor — 0=audio degraded, 1=video degraded, 2=both clean

        Returns:
            Scalar loss
        """
        # Vectorized: build per-sample targets via masking instead of a Python loop.
        # corruption_type == 0 → target = video_proj
        # corruption_type == 1 → target = audio_proj
        # corruption_type == 2 → target = average of both
        is_audio_deg = (corruption_type == 0).unsqueeze(1).float()  # (n, 1)
        is_video_deg = (corruption_type == 1).unsqueeze(1).float()  # (n, 1)
        is_clean     = (corruption_type == 2).unsqueeze(1).float()  # (n, 1)

        target = (
            is_audio_deg * video_proj
            + is_video_deg * audio_proj
            + is_clean * (audio_proj + video_proj) / 2
        )

        return self.mse(fused_embed, target)


class QualityAlignmentLoss(nn.Module):
    """
    Gating weights should correlate with the known corruption levels.
    Audio weight should be proportional to audio quality, and vice versa.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        audio_weight: torch.Tensor,
        audio_quality: torch.Tensor,
        video_quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_weight: (n,) gating network's audio weight output
            audio_quality: (n, 2) [snr_norm, vad_conf]
            video_quality: (n, 3) [blur_norm, exposure_norm, det_stability]

        Returns:
            Scalar loss
        """
        # Expected audio weight derived from quality signals
        expected_audio = torch.clamp(audio_quality[:, 0], 0.1, 0.9)
        expected_video = torch.clamp(video_quality[:, 0], 0.1, 0.9)
        total = expected_audio + expected_video
        expected_audio = expected_audio / (total + 1e-8)

        return self.mse(audio_weight, expected_audio)


class EntropyRegularization(nn.Module):
    """
    Prevents weight collapse.

    Penalizes only weights whose entropy falls below target_entropy.
    Allows decisive weights (e.g., 0.9/0.1) while discouraging saturation at 0/1.
    """

    def __init__(self, target_entropy: float = 0.5):
        """
        Args:
            target_entropy: minimum acceptable entropy.
                            0 = saturated, ln 2 ≈ 0.693 = uniform.
                            Default 0.5 allows moderate decisiveness.
        """
        super().__init__()
        self.target_entropy = target_entropy

    def forward(
        self,
        audio_weight: torch.Tensor,
        video_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_weight: (n,) audio modality weight
            video_weight: (n,) video modality weight

        Returns:
            Scalar loss — penalizes only when entropy < target_entropy
        """
        entropy = -(
            audio_weight * torch.log(audio_weight + 1e-8)
            + video_weight * torch.log(video_weight + 1e-8)
        )
        below_target = torch.clamp(self.target_entropy - entropy, min=0.0)
        return below_target.mean()
