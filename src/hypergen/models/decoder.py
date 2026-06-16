"""Convolutional decoder mapping latent codes on S^{d-1} to 32x32x3 images."""

from __future__ import annotations

import torch
from torch import nn


class ConvDecoder(nn.Module):
    """Projects a latent code to a 32x32x3 image (logits).

    Args:
        latent_dim: Dimensionality d of the input latent code.
        feature_dim: Width of the initial projected feature map channel count.
    """

    def __init__(self, latent_dim: int = 64, feature_dim: int = 256) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, feature_dim * 4 * 4)
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project(z).view(z.shape[0], self.feature_dim, 4, 4)
        return self.net(h)
