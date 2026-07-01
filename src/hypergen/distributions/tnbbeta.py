"""TNBbeta distribution on the unit interval.

Implements the density, exact (rejection-free) reparameterized sampling via the
``T_p \\circ H_q`` transform of a ``Beta(eps, eps)`` base sample, and a
Gauss-Legendre quadrature helper for the normalizing constant used in the
analytic KL term of the TNBbeta-VAE.

Reference:
    Lederman & Schein (2026), "The Triply-Randomized Negative Binomial Beta
    for Robust Regression and Conjugate Models of Bounded Support Data":
    https://arxiv.org/abs/2606.11624
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import Beta, Distribution, constraints

_EPS_CLAMP = 1e-6


def _logit(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable logit on (0, 1)."""
    y = y.clamp(_EPS_CLAMP, 1.0 - _EPS_CLAMP)
    return torch.log(y) - torch.log1p(-y)


def _log_sech(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(sech(x)) = -log(cosh(x))."""
    abs_x = x.abs()
    return math.log(2.0) - abs_x - torch.log1p(torch.exp(-2.0 * abs_x))


def _log_gamma(y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """log gamma(y, p) = log(1/4) + 2 * log(sech((logit(y) - logit(p)) / 2))."""
    d = (_logit(y) - _logit(p)) / 2.0
    return -math.log(4.0) + 2.0 * _log_sech(d)


class TNBbeta(Distribution):
    """The TNBbeta distribution on (0, 1).

    Args:
        p: Location parameter, strictly in (0, 1).
        q: Concentration/heavy-tail parameter, in [0, 1/4).
        eps: Shape parameter (inherited from the base Beta(eps, eps)), > 0.
    """

    arg_constraints = {
        "p": constraints.interval(0.0, 1.0),
        "q": constraints.interval(0.0, 1.0),
        "eps": constraints.positive,
    }
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.p, self.q, self.eps = torch.broadcast_tensors(p, q, eps)
        batch_shape = self.p.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _base_beta(self) -> Beta:
        return Beta(self.eps, self.eps)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        shape = torch.Size(sample_shape)
        eps = self.eps.expand(shape + self.eps.shape)
        p = self.p.expand(shape + self.p.shape)
        q = self.q.expand(shape + self.q.shape)

        z = Beta(eps, eps).rsample()
        s = 2.0 * z - 1.0
        u = 0.5 * (1.0 + s * torch.sqrt((1.0 - q) / (1.0 - q * s.pow(2))))
        y = p * u / ((1.0 - p) * (1.0 - u) + p * u)
        return y.clamp(_EPS_CLAMP, 1.0 - _EPS_CLAMP)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        y = value.clamp(_EPS_CLAMP, 1.0 - _EPS_CLAMP)
        p, q, eps = self.p, self.q, self.eps

        # log beta(y; eps, eps) = (eps-1) * [log(y) + log(1-y)] - log B(eps, eps)
        log_beta_fn = torch.lgamma(eps) * 2.0 - torch.lgamma(2.0 * eps)
        log_base = (eps - 1.0) * (torch.log(y) + torch.log1p(-y)) - log_beta_fn

        log_gamma = _log_gamma(y, p)
        log_term1 = eps * (torch.log(1.0 - q) - torch.log(y) - torch.log1p(-y) + log_gamma)
        term2 = 1.0 - 4.0 * q * torch.exp(log_gamma)
        term2 = term2.clamp_min(_EPS_CLAMP)
        log_term2 = -(eps + 0.5) * torch.log(term2)

        return log_base + log_term1 + log_term2

    def wasserstein1(
        self,
        prior: Distribution,
        n_samples: int = 64,
    ) -> torch.Tensor:
        """W_1 between self and prior via sorted-sample approximation.

        For 1D distributions W_1 = integral |F^{-1}(u) - G^{-1}(u)| du,
        which equals the L1 distance between sorted sample vectors.
        """
        y_post = self.rsample(torch.Size([n_samples]))  # (K, *batch)
        y_prior = prior.rsample(torch.Size([n_samples] + list(self.batch_shape)))  # (K, *batch)
        y_post_sorted = y_post.sort(dim=0).values
        y_prior_sorted = y_prior.sort(dim=0).values
        return (y_post_sorted - y_prior_sorted).abs().mean(dim=0)

    def normalizing_constant(self, n_points: int = 32) -> torch.Tensor:
        """Gauss-Legendre quadrature estimate of integral of the density over (0, 1).

        Used as a sanity check / correction factor in the analytic KL term.
        Should be close to 1.0 for a correctly normalized density.

        Args:
            n_points: Number of quadrature points (32 is sufficient per spec).

        Returns:
            Tensor broadcasting over the batch shape, the estimated integral.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        # Map nodes from [-1, 1] to (0, 1).
        y_nodes = torch.as_tensor(0.5 * (nodes + 1.0), dtype=self.p.dtype, device=self.p.device)
        w_nodes = torch.as_tensor(0.5 * weights, dtype=self.p.dtype, device=self.p.device)

        total = torch.zeros_like(self.p)
        for y_i, w_i in zip(y_nodes, w_nodes, strict=True):
            y_b = y_i.expand_as(self.p)
            total = total + w_i * torch.exp(self.log_prob(y_b))
        return total
