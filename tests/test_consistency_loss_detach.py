"""CrossModalConsistencyLoss must not train the projections it regresses against.

Without .detach() the projections are nn.Linear weights, so the loss has a
trivial optimum: scale both toward zero, MSE -> 0, fused_embed never has to
match anything. The gradient check below fails the moment detach is removed.
"""

import torch

from src.training.losses import CrossModalConsistencyLoss


def test_no_gradient_flows_into_the_target_projections():
    torch.manual_seed(0)
    audio_proj = torch.nn.Linear(16, 8)
    video_proj = torch.nn.Linear(16, 8)
    fused_embed = torch.nn.Linear(16, 8)

    loss = CrossModalConsistencyLoss()
    n = 4
    corruption = torch.tensor([0, 1, 2, 2])

    loss(
        fused_embed(torch.randn(n, 16)),
        audio_proj(torch.randn(n, 16)),
        video_proj(torch.randn(n, 16)),
        corruption,
    ).backward()

    for name, proj in (("audio_proj", audio_proj), ("video_proj", video_proj)):
        grads = [p.grad for p in proj.parameters()]
        assert all(g is None or torch.count_nonzero(g) == 0 for g in grads), (
            f"{name} received gradient from the consistency loss — the "
            "projections must be a detached teacher, else collapsing them "
            "to zero minimises the loss without training the fusion model"
        )

    # The thing we DO want to train must still get gradient.
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in fused_embed.parameters()
    ), "fused_embed path is dead — detach went too far"
