"""
Train the quality-aware gating network.

Uses synthetic corruption data to train the gating network to:
1. Produce modality weights that track quality signals
2. Fuse embeddings closer to the uncorrupted modality
3. Maintain moderate entropy (not collapse to 0.5/0.5 or 1.0/0.0)

Usage:
    python scripts/train_gating.py --config config/training.yaml
    python scripts/train_gating.py --epochs 50 --lr 5e-4
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def generate_synthetic_data(
    n_samples: int = 200,
    embed_dim: int = 1024,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic multimodal embedding pairs with quality labels.

    Returns:
        Dict with audio_embeds, video_embeds, audio_quality, video_quality,
        corruption_type arrays
    """
    rng = np.random.RandomState(seed)

    # Base embeddings (random but normalized)
    audio_embeds = rng.randn(n_samples, embed_dim).astype(np.float32)
    video_embeds = rng.randn(n_samples, embed_dim).astype(np.float32)
    audio_embeds /= np.linalg.norm(audio_embeds, axis=1, keepdims=True) + 1e-8
    video_embeds /= np.linalg.norm(video_embeds, axis=1, keepdims=True) + 1e-8

    # Corruption types: 0=audio degraded, 1=video degraded, 2=both clean
    corruption_type = np.zeros(n_samples, dtype=int)
    corruption_type[:n_samples // 3] = 0
    corruption_type[n_samples // 3: 2 * n_samples // 3] = 1
    corruption_type[2 * n_samples // 3:] = 2

    # Quality scores correlate with corruption
    audio_quality = np.zeros((n_samples, 2), dtype=np.float32)
    video_quality = np.zeros((n_samples, 3), dtype=np.float32)

    for i in range(n_samples):
        if corruption_type[i] == 0:  # audio degraded
            audio_quality[i] = [0.1 + rng.uniform(0, 0.2), 0.2 + rng.uniform(0, 0.1)]
            video_quality[i] = [0.7 + rng.uniform(0, 0.2), 0.7, 0.8 + rng.uniform(0, 0.1)]
        elif corruption_type[i] == 1:  # video degraded
            audio_quality[i] = [0.7 + rng.uniform(0, 0.2), 0.7 + rng.uniform(0, 0.1)]
            video_quality[i] = [0.1 + rng.uniform(0, 0.2), 0.7, 0.1 + rng.uniform(0, 0.1)]
        else:  # both clean
            audio_quality[i] = [0.7 + rng.uniform(0, 0.2), 0.7 + rng.uniform(0, 0.2)]
            video_quality[i] = [0.7 + rng.uniform(0, 0.2), 0.7, 0.7 + rng.uniform(0, 0.2)]

    # Degrade embeddings based on corruption type
    noise_rng = np.random.RandomState(seed + 1)
    for i in range(n_samples):
        if corruption_type[i] == 0:  # audio degraded
            noise_level = 1.0 - audio_quality[i, 0]
            audio_embeds[i] += noise_rng.randn(embed_dim).astype(np.float32) * noise_level
        elif corruption_type[i] == 1:  # video degraded
            noise_level = 1.0 - video_quality[i, 0]
            video_embeds[i] += noise_rng.randn(embed_dim).astype(np.float32) * noise_level

    return {
        "audio_embeds": audio_embeds,
        "video_embeds": video_embeds,
        "audio_quality": audio_quality,
        "video_quality": video_quality,
        "corruption_type": corruption_type,
    }


def train_epoch(
    model: QualityAwareFusion,
    data: dict,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_weights: dict,
) -> dict:
    """Train for one epoch. Returns loss dict."""
    model.train()

    audio_embeds = torch.from_numpy(data["audio_embeds"]).float().to(device)
    video_embeds = torch.from_numpy(data["video_embeds"]).float().to(device)
    audio_q = torch.from_numpy(data["audio_quality"]).float().to(device)
    video_q = torch.from_numpy(data["video_quality"]).float().to(device)
    corruption = torch.from_numpy(data["corruption_type"]).long().to(device)

    optimizer.zero_grad()

    result = model(
        audio_embeds, video_embeds,
        audio_q, video_q,
        use_dynamic_weights=True,
    )
    fused = result["fused_embed"]
    w_audio = result["audio_weight"]
    w_video = result["video_weight"]

    # Project embeddings for consistency loss
    audio_proj = model.fusion.audio_proj(audio_embeds)
    video_proj = model.fusion.video_proj(video_embeds)

    # Compute losses
    l_consistency = CrossModalConsistencyLoss()(fused, audio_proj, video_proj, corruption)
    l_quality = QualityAlignmentLoss()(w_audio, audio_q, video_q)
    l_entropy = EntropyRegularization()(w_audio, w_video)

    total_loss = (
        loss_weights["lambda_consistency"] * l_consistency
        + loss_weights["lambda_quality"] * l_quality
        + loss_weights["lambda_entropy"] * l_entropy
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total": total_loss.item(),
        "consistency": l_consistency.item(),
        "quality": l_quality.item(),
        "entropy": l_entropy.item(),
        "mean_audio_weight": w_audio.mean().item(),
        "mean_video_weight": w_video.mean().item(),
    }


@torch.no_grad()
def evaluate(
    model: QualityAwareFusion,
    data: dict,
    device: str,
    loss_weights: dict,
) -> dict:
    """Evaluate on validation data. Returns loss dict + metrics."""
    model.eval()

    audio_embeds = torch.from_numpy(data["audio_embeds"]).float().to(device)
    video_embeds = torch.from_numpy(data["video_embeds"]).float().to(device)
    audio_q = torch.from_numpy(data["audio_quality"]).float().to(device)
    video_q = torch.from_numpy(data["video_quality"]).float().to(device)
    corruption = torch.from_numpy(data["corruption_type"]).long().to(device)

    result = model(
        audio_embeds, video_embeds,
        audio_q, video_q,
        use_dynamic_weights=True,
    )
    fused = result["fused_embed"]
    w_audio = result["audio_weight"]
    w_video = result["video_weight"]

    audio_proj = model.fusion.audio_proj(audio_embeds)
    video_proj = model.fusion.video_proj(video_embeds)

    l_consistency = CrossModalConsistencyLoss()(fused, audio_proj, video_proj, corruption)
    l_quality = QualityAlignmentLoss()(w_audio, audio_q, video_q)
    l_entropy = EntropyRegularization()(w_audio, w_video)

    total_loss = (
        loss_weights["lambda_consistency"] * l_consistency
        + loss_weights["lambda_quality"] * l_quality
        + loss_weights["lambda_entropy"] * l_entropy
    )

    # Degradation response: weight difference between audio-degraded and video-degraded
    audio_deg_mask = corruption == 0
    video_deg_mask = corruption == 1
    if audio_deg_mask.sum() > 0 and video_deg_mask.sum() > 0:
        w_audio_when_audio_deg = w_audio[audio_deg_mask].mean().item()
        w_audio_when_video_deg = w_audio[video_deg_mask].mean().item()
        degradation_response = abs(w_audio_when_audio_deg - w_audio_when_video_deg)
    else:
        degradation_response = 0.0

    return {
        "total": total_loss.item(),
        "consistency": l_consistency.item(),
        "quality": l_quality.item(),
        "entropy": l_entropy.item(),
        "degradation_response": degradation_response,
        "mean_audio_weight": w_audio.mean().item(),
        "mean_video_weight": w_video.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train quality-aware gating network")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(train_cfg["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate data
    logger.info(f"Generating {args.n_samples} synthetic training samples...")
    data = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)

    # Split data
    n = args.n_samples
    n_train = int(n * train_cfg["train_split"])
    n_val = int(n * train_cfg["val_split"])

    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_data = {k: v[train_idx] if isinstance(v, np.ndarray) else v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] if isinstance(v, np.ndarray) else v[val_idx] for k, v in data.items()}

    logger.info(f"Train: {n_train}, Val: {n_val}")

    # Initialize model
    fusion_cfg = train_cfg["fusion"]
    model = QualityAwareFusion(
        audio_dim=fusion_cfg["audio_dim"],
        video_dim=fusion_cfg["video_dim"],
        hidden_dim=fusion_cfg["hidden_dim"],
        num_heads=fusion_cfg["num_heads"],
        num_layers=fusion_cfg["num_layers"],
        dropout=fusion_cfg["dropout"],
        use_learned_gating=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    train_cfg["learning_rate"] = float(train_cfg["learning_rate"])
    train_cfg["weight_decay"] = float(train_cfg["weight_decay"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"]
    )

    loss_weights = {
        "lambda_consistency": float(train_cfg["lambda_consistency"]),
        "lambda_quality": float(train_cfg["lambda_quality"]),
        "lambda_entropy": float(train_cfg["lambda_entropy"]),
    }

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    logger.info("Starting training...")
    for epoch in range(train_cfg["epochs"]):
        t0 = time.monotonic()

        train_metrics = train_epoch(model, train_data, optimizer, device, loss_weights)
        val_metrics = evaluate(model, val_data, device, loss_weights)

        dt = time.monotonic() - t0

        # Learning rate warmup
        if epoch < train_cfg["warmup_epochs"]:
            warmup_factor = (epoch + 1) / train_cfg["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg["learning_rate"] * warmup_factor
        else:
            scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": dt,
        }
        history.append(record)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{train_cfg['epochs']} ({dt:.1f}s): "
                f"train_loss={train_metrics['total']:.4f}, "
                f"val_loss={val_metrics['total']:.4f}, "
                f"degradation_response={val_metrics['degradation_response']:.4f}, "
                f"w_audio={val_metrics['mean_audio_weight']:.3f}"
            )

        # Early stopping
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0
            # Save best model
            torch.save(
                model.state_dict(),
                output_dir / f"gating_best_{timestamp}.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Periodic checkpoint
        if (epoch + 1) % train_cfg["save_every"] == 0:
            torch.save(
                model.state_dict(),
                output_dir / f"gating_epoch{epoch+1}_{timestamp}.pt",
            )

    # Save training history
    history_path = output_dir / f"training_history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save(
        model.state_dict(),
        output_dir / f"gating_final_{timestamp}.pt",
    )

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"History saved to {history_path}")
    logger.info(f"Models saved to {output_dir}")


if __name__ == "__main__":
    main()
