"""Tests for latent-space metrics."""

from __future__ import annotations

import torch

from hypergen.metrics.isotropy import isotropy_score
from hypergen.metrics.probes import LinearProbe


def test_isotropy_score_uniform() -> None:
    torch.manual_seed(0)
    dim = 16
    points = torch.randn(20_000, dim)
    points = points / points.norm(dim=-1, keepdim=True)
    score = isotropy_score(points)
    assert score.item() < 0.05


def test_isotropy_score_collapsed() -> None:
    dim = 16
    mu = torch.zeros(dim)
    mu[0] = 1.0
    points = mu.expand(1000, dim)
    score = isotropy_score(points)
    assert score.item() > 0.5


def test_linear_probe_shapes() -> None:
    probe = LinearProbe(latent_dim=32, num_classes=100)
    z = torch.randn(8, 32)
    logits = probe(z)
    assert logits.shape == (8, 100)
