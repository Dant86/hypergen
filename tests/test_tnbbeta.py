"""Tests for the TNBbeta distribution."""

from __future__ import annotations

import itertools

import pytest
import torch

from hypergen.distributions.tnbbeta import TNBbeta

P_VALUES = [0.2, 0.5, 0.8]
Q_VALUES = [0.0, 0.1, 0.2]
EPS_VALUES = [0.5, 1.0, 5.0]


def _make(p: float, q: float, eps: float) -> TNBbeta:
    return TNBbeta(torch.tensor(p), torch.tensor(q), torch.tensor(eps))


def test_density_positive() -> None:
    dist = _make(0.5, 0.1, 1.0)
    y = torch.linspace(0.01, 0.99, 200)
    density = torch.exp(dist.log_prob(y))
    assert torch.all(density > 0)


@pytest.mark.parametrize("p,q,eps", list(itertools.product(P_VALUES, Q_VALUES, EPS_VALUES)))
def test_density_integrates_to_one(p: float, q: float, eps: float) -> None:
    dist = _make(p, q, eps)
    # eps < 1 gives a Beta(eps, eps) base with singularities at the boundary
    # (e.g. the arcsine distribution at eps=0.5), which needs many more
    # quadrature points to resolve than the well-behaved eps >= 1 cases.
    n_points = 512 if eps < 1.0 else 64
    integral = dist.normalizing_constant(n_points=n_points)
    assert torch.allclose(integral, torch.tensor(1.0), atol=1e-2)


def test_reparameterization_samples_in_unit_interval() -> None:
    dist = _make(0.5, 0.1, 1.0)
    samples = dist.rsample(torch.Size([10_000]))
    assert torch.all(samples > 0.0)
    assert torch.all(samples < 1.0)


def test_gradient_flows() -> None:
    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    eps = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    torch.manual_seed(0)

    def sample_fn(p: torch.Tensor, q: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        return TNBbeta(p, q, eps).rsample()

    # Beta.rsample() uses an implicit-reparameterization gradient estimator,
    # which is only approximate (vs. exact pathwise derivatives), so a looser
    # tolerance is needed than for closed-form reparameterizations.
    assert torch.autograd.gradcheck(sample_fn, (p, q, eps), eps=1e-6, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("p_val", [0.2, 0.35, 0.5, 0.65, 0.8])
def test_mode_at_p(p_val: float) -> None:
    dist = _make(p_val, 0.0, 5.0)
    samples = dist.rsample(torch.Size([20_000]))
    empirical_mode = samples.median().item()
    assert abs(empirical_mode - p_val) < 0.1
