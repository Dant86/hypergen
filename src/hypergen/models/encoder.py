"""Encoder producing TNBbeta-Spherical posterior parameters (mu, p, q, eps)."""

from __future__ import annotations

import torch
from torch import nn

_P_CLAMP = 2e-4
_Q_CLAMP = 1e-5
_EPS_FLOOR = 1e-3


class ConvBackbone(nn.Module):
    """Lightweight ConvNet backbone for 32x32x3 images."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, 4, stride=1, padding=0),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class TNBbetaEncoder(nn.Module):
    """Encoder mapping images to (mu, p, q, eps) for the TNBbeta-Spherical posterior.

    Args:
        latent_dim: Dimensionality d of the latent sphere S^{d-1}.
        feature_dim: Width of the penultimate feature vector.
    """

    def __init__(self, latent_dim: int = 64, feature_dim: int = 512) -> None:
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim)
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.p_head = nn.Linear(feature_dim, 1)
        self.q_head = nn.Linear(feature_dim, 1)
        self.eps_head = nn.Linear(feature_dim, 1)
        self.latent_dim = latent_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)

        mu_raw = self.mu_head(feats)
        mu = mu_raw / mu_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        p = torch.sigmoid(self.p_head(feats).squeeze(-1)) * (1.0 - 2.0 * _P_CLAMP) + _P_CLAMP
        q = (torch.sigmoid(self.q_head(feats).squeeze(-1)) / 4.0 - _Q_CLAMP).clamp_min(0.0)
        eps = nn.functional.softplus(self.eps_head(feats).squeeze(-1)) + _EPS_FLOOR

        return mu, p, q, eps
