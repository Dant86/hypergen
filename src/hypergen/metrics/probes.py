"""Linear probe accuracy and active-unit count on latent codes."""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    """A single linear classifier fit on frozen latent codes.

    Args:
        latent_dim: Dimensionality of the input latent codes.
        num_classes: Number of output classes (100 for CIFAR-100).
    """

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


def active_units(mu_per_example: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Counts latent dimensions with Var_x[E_z[z_i]] > threshold.

    Args:
        mu_per_example: Posterior mean directions, shape (N, d).
        threshold: Activity threshold (default 0.01, per spec).

    Returns:
        Boolean tensor of shape (d,), True where the unit is active.
    """
    variances = mu_per_example.var(dim=0, unbiased=True)
    return variances > threshold
