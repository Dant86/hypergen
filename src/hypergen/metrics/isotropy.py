"""Isotropy of the aggregate posterior on S^{d-1}."""

from __future__ import annotations

import torch


def isotropy_score(mus: torch.Tensor) -> torch.Tensor:
    """Computes ||A - (1/d) I||_F where A = (1/N) sum_i mu_i mu_i^T.

    Args:
        mus: Encoder mean directions, shape (N, d), assumed unit norm.

    Returns:
        Scalar Frobenius-norm isotropy score. Lower is more isotropic.
    """
    n, d = mus.shape
    a = (mus.T @ mus) / n
    target = torch.eye(d, dtype=mus.dtype, device=mus.device) / d
    return torch.linalg.norm(a - target, ord="fro")
