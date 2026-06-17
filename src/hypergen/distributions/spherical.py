"""Lift of the TNBbeta distribution to the unit hypersphere S^{d-1}.

Construction mirrors Power Spherical (De Cao & Aziz, 2020): sample the
cosine-similarity coordinate from TNBbeta, sample the orthogonal component
uniformly on S^{d-2}, then rotate the north-pole construction onto the mode
direction mu via a Householder reflection.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import Distribution

from hypergen.distributions.tnbbeta import TNBbeta

_EPS_CLAMP = 1e-6


def _uniform_on_sphere(
    shape: torch.Size, dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Draws uniform samples on S^{dim-1} embedded in R^dim."""
    v = torch.randn(*shape, dim, dtype=dtype, device=device)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(_EPS_CLAMP)


def householder_matrix(mu: torch.Tensor) -> torch.Tensor:
    """Builds the Householder reflection H mapping e_1 to mu, batched.

    Args:
        mu: Tensor of shape (..., d), unit vectors.

    Returns:
        Tensor of shape (..., d, d).
    """
    d = mu.shape[-1]
    e1 = torch.zeros_like(mu)
    e1[..., 0] = 1.0
    u = e1 - mu
    u_norm = u.norm(dim=-1, keepdim=True)
    # Where mu == e1 (u_norm ~ 0), the reflection should be the identity.
    safe_norm = u_norm.clamp_min(_EPS_CLAMP)
    u_hat = u / safe_norm
    eye = torch.eye(d, dtype=mu.dtype, device=mu.device).expand(*mu.shape[:-1], d, d)
    outer = u_hat.unsqueeze(-1) * u_hat.unsqueeze(-2)
    house = eye - 2.0 * outer
    is_identity = (u_norm < 10.0 * _EPS_CLAMP).unsqueeze(-1)
    return torch.where(is_identity, eye, house)


class TNBbetaSpherical(Distribution):
    """TNBbeta lifted onto S^{d-1} via the Householder construction.

    Args:
        mu: Mode direction, unit vectors of shape (..., d).
        p: TNBbeta location parameter, shape (...).
        q: TNBbeta concentration parameter, shape (...).
        eps: TNBbeta shape parameter, shape (...).
    """

    arg_constraints: dict[str, object] = {}
    has_rsample = True

    def __init__(
        self,
        mu: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        eps: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.mu = mu
        self.dim = mu.shape[-1]
        self.tnbbeta = TNBbeta(p, q, eps, validate_args=validate_args)
        batch_shape = mu.shape[:-1]
        super().__init__(batch_shape=batch_shape, validate_args=False)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        shape = torch.Size(sample_shape)
        y = self.tnbbeta.rsample(shape)  # shape: sample_shape + batch_shape
        s = (2.0 * y - 1.0).clamp(-1.0 + _EPS_CLAMP, 1.0 - _EPS_CLAMP)

        v = _uniform_on_sphere(
            shape + self.batch_shape, self.dim - 1, self.mu.dtype, self.mu.device
        )
        ortho_scale = torch.sqrt((1.0 - s.pow(2)).clamp_min(0.0)).unsqueeze(-1)
        x_north = torch.cat([s.unsqueeze(-1), ortho_scale * v], dim=-1)

        mu_expanded = self.mu.expand(shape + self.mu.shape)
        house = householder_matrix(mu_expanded)
        x = torch.einsum("...ij,...j->...i", house, x_north)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(_EPS_CLAMP)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        with torch.no_grad():
            return self.rsample(sample_shape)

    def alignment(self, x: torch.Tensor) -> torch.Tensor:
        """R = (1 + mu^T x) / 2, the TNBbeta-marginal coordinate of x."""
        cos_sim = (self.mu * x).sum(dim=-1)
        return ((1.0 + cos_sim) / 2.0).clamp(_EPS_CLAMP, 1.0 - _EPS_CLAMP)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Full density of x on S^{d-1}.

        log p(x) = log TNBbeta(R(x)) + (d-3)/2 * log(1 - (2R-1)^2)
                   - log(surface_area(S^{d-2})) - log(2)
        """
        r = self.alignment(value)
        d = self.dim
        jacobian_term = (
            (d - 3) / 2.0 * torch.log((1.0 - (2.0 * r - 1.0).pow(2)).clamp_min(_EPS_CLAMP))
        )
        log_surface_d_minus_2 = (
            math.log(2.0)
            + ((d - 1) / 2.0) * math.log(math.pi)
            - torch.lgamma(torch.tensor((d - 1) / 2.0, dtype=value.dtype, device=value.device))
        )
        return self.tnbbeta.log_prob(r) + jacobian_term - log_surface_d_minus_2 - math.log(2.0)

    @staticmethod
    def log_uniform_density(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """log density of Uniform(S^{dim-1}) = -log(surface area of S^{dim-1})."""
        log_surface_area = (
            math.log(2.0)
            + (dim / 2.0) * math.log(math.pi)
            - torch.lgamma(torch.tensor(dim / 2.0, dtype=dtype, device=device))
        )
        return -log_surface_area
