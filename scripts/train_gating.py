"""
Train the Layer-2a quality-aware gating/fusion network.

Trains `QualityAwareFusion` (src/layer2/fusion.py) with the three objectives in
src/training/losses.py:
  1. CrossModalConsistencyLoss — fused embed tracks the cleaner modality
  2. QualityAlignmentLoss      — gate weights track modality quality
  3. EntropyRegularization     — no collapse to 0/1

There is no labeled video set with per-modality quality annotations in the repo,
so this bootstraps from synthetic (quality, corruption) pairs. Swap
`_synthesize_batch` for a real loader once one exists — the training loop and
checkpoint format do not change. The resulting checkpoint is consumed by
config layer2a.fusion.checkpoint, and the pipeline then sets gating_trained=True
so the sanity check stops overriding the learned gate.

ponytail: synthetic bootstrap only; real (quality, label) video data is the
upgrade path. Trained here so the gate has a defined, reproducible artifact.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer2.fusion import QualityAwareFusion  # noqa: E402
from src.training.losses import (  # noqa: E402
    CrossModalConsistencyLoss,
    EntropyRegularization,
    QualityAlignmentLoss,
)


def _config() -> dict:
    import yaml
    path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return yaml.safe_load(path.read_text()) or {}


def _synthesize_batch(batch_size, audio_dim, video_dim, device, generator):
    """Random embeddings + quality signals with quality-correlated corruption.

    0 = audio degraded (low audio quality), 1 = video degraded, 2 = both clean.
    """
    audio = torch.randn(batch_size, audio_dim, generator=generator, device=device)
    video = torch.randn(batch_size, video_dim, generator=generator, device=device)
    audio_q = torch.rand(batch_size, 2, generator=generator, device=device)
    video_q = torch.rand(batch_size, 3, generator=generator, device=device)
    audio_scalar = audio_q.mean(dim=1)
    video_scalar = video_q.mean(dim=1)
    corruption = torch.where(
        audio_scalar < video_scalar,
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.ones(batch_size, dtype=torch.long, device=device),
    )
    # A clean slice, so the loss sees all three cases.
    clean = torch.rand(batch_size, generator=generator, device=device) < 0.25
    corruption = torch.where(
        clean, torch.full_like(corruption, 2), corruption)
    return audio, video, audio_q, video_q, corruption


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="weights/fusion_gating.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = _config()
    l1 = cfg.get("layer1", {})
    audio_dim = int(l1.get("audio_events", {}).get("embedding_dim", 256))
    video_dim = int(l1.get("visual_embeddings", {}).get("output_dim", 768))
    fc = cfg.get("layer2a", {}).get("fusion", {})

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    model = QualityAwareFusion(
        audio_dim=audio_dim, video_dim=video_dim,
        hidden_dim=int(fc.get("hidden_dim", 512)),
        num_heads=int(fc.get("num_heads", 8)),
        num_layers=int(fc.get("num_layers", 3)),
        dropout=float(fc.get("dropout", 0.1)),
        use_learned_gating=True,
    ).to(device)

    consistency = CrossModalConsistencyLoss()
    alignment = QualityAlignmentLoss()
    entropy = EntropyRegularization()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            batch = _synthesize_batch(
                args.batch_size, audio_dim, video_dim, device, gen)
            audio, video, aq, vq, corruption = batch
            out = model(audio, video, audio_quality=aq, video_quality=vq,
                        use_dynamic_weights=True)
            audio_h = model.fusion.audio_proj(audio)
            video_h = model.fusion.video_proj(video)
            loss = (
                consistency(out["fused_embed"], audio_h, video_h, corruption)
                + alignment(out["audio_weight"], aq, vq)
                + 0.1 * entropy(out["audio_weight"], out["video_weight"])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{args.epochs}  "
                  f"loss {epoch_loss / args.steps_per_epoch:.4f}")

    model.gating_trained = True
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved trained gating/fusion checkpoint -> {out_path}")
    print("set layer2a.fusion.checkpoint to this path to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
