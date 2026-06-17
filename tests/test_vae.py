"""Tests for the TNBbeta-VAE."""

from __future__ import annotations

import torch

from hypergen.models.vae import TNBbetaVAE

LATENT_DIM = 16


def _random_batch(batch_size: int = 4) -> torch.Tensor:
    return torch.rand(batch_size, 3, 32, 32)


def test_elbo_is_finite() -> None:
    model = TNBbetaVAE(latent_dim=LATENT_DIM)
    out = model.forward(_random_batch())
    assert torch.isfinite(out.elbo)
    assert torch.isfinite(out.recon_loss)
    assert torch.isfinite(out.kl)


def test_elbo_decreases() -> None:
    torch.manual_seed(0)
    model = TNBbetaVAE(latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = _random_batch()

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = model.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Compare smoothed early vs. late loss to absorb sampling noise from the
    # stochastic KL/reconstruction terms rather than relying on a single step.
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early


def test_kl_nonnegative() -> None:
    model = TNBbetaVAE(latent_dim=LATENT_DIM)
    out = model.forward(_random_batch())
    assert out.kl.item() >= 0.0


def test_encoder_output_constraints() -> None:
    model = TNBbetaVAE(latent_dim=LATENT_DIM)
    _mu, p, q, eps = model.encoder(_random_batch(16))
    assert torch.all(p > 0.0) and torch.all(p < 1.0)
    assert torch.all(q >= 0.0) and torch.all(q < 0.25)
    assert torch.all(eps > 0.0)
