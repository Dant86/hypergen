"""Tests for the spherical lift of TNBbeta."""

from __future__ import annotations

from scipy import stats
import torch

from hypergen.distributions.spherical import TNBbetaSpherical
from hypergen.distributions.tnbbeta import TNBbeta


def _make_mu(dim: int) -> torch.Tensor:
    mu = torch.zeros(dim)
    mu[0] = 1.0
    return mu


def test_samples_on_sphere() -> None:
    dim = 8
    mu = _make_mu(dim)
    dist = TNBbetaSpherical(mu, torch.tensor(0.5), torch.tensor(0.1), torch.tensor(1.0))
    samples = dist.rsample(torch.Size([1000]))
    norms = samples.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_alignment_distribution() -> None:
    dim = 8
    mu = _make_mu(dim)
    p, q, eps = 0.5, 0.1, 2.0
    dist = TNBbetaSpherical(mu, torch.tensor(p), torch.tensor(q), torch.tensor(eps))
    samples = dist.rsample(torch.Size([5000]))
    r = dist.alignment(samples).numpy()

    base = TNBbeta(torch.tensor(p), torch.tensor(q), torch.tensor(eps))
    reference = base.rsample(torch.Size([5000])).numpy()

    result = stats.ks_2samp(r, reference)
    assert result.pvalue > 0.05  # type: ignore[attr-defined]


def test_householder_equivariance() -> None:
    dim = 8
    torch.manual_seed(0)
    rotation = torch.linalg.qr(torch.randn(dim, dim))[0]

    mu1 = _make_mu(dim)
    mu2 = rotation @ mu1

    torch.manual_seed(42)
    dist1 = TNBbetaSpherical(mu1, torch.tensor(0.5), torch.tensor(0.1), torch.tensor(1.0))
    samples1 = dist1.rsample(torch.Size([1000]))

    torch.manual_seed(42)
    dist2 = TNBbetaSpherical(mu2, torch.tensor(0.5), torch.tensor(0.1), torch.tensor(1.0))
    samples2 = dist2.rsample(torch.Size([1000]))

    rotated_samples1 = samples1 @ rotation.T
    # Same underlying randomness rotated should match the directly-rotated-mu samples
    # in distribution: compare mean cosine similarity to mu as a coarse equivariance check.
    cos1 = (rotated_samples1 * mu2).sum(dim=-1).mean()
    cos2 = (samples2 * mu2).sum(dim=-1).mean()
    assert torch.allclose(cos1, cos2, atol=0.1)
