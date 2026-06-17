"""Standard Gaussian VAE baseline, sharing the encoder/decoder backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from hypergen.models.decoder import ConvDecoder
from hypergen.models.encoder import ConvBackbone


@dataclass
class GaussianELBOOutput:
    """Container for the components of the Gaussian-VAE ELBO."""

    elbo: torch.Tensor
    recon_loss: torch.Tensor
    kl: torch.Tensor
    recon: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor


class GaussianVAE(nn.Module):
    """VAE with N(mu, sigma^2 I) posterior and a standard normal prior."""

    def __init__(self, latent_dim: int = 64, beta: float = 1.0, feature_dim: int = 512) -> None:
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim)
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.log_sigma_head = nn.Linear(feature_dim, latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        mu = self.mu_head(feats)
        log_sigma = self.log_sigma_head(feats).clamp(-10.0, 10.0)
        return mu, log_sigma

    def forward(self, x: torch.Tensor) -> GaussianELBOOutput:
        mu, log_sigma = self.encode(x)
        posterior = Normal(mu, log_sigma.exp())
        prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))

        z = posterior.rsample()
        recon = self.decoder(z)
        recon_loss = (
            nn.functional.binary_cross_entropy_with_logits(recon, x, reduction="none")
            .flatten(1)
            .sum(dim=1)
        )

        kl = kl_divergence(posterior, prior).sum(dim=-1)
        elbo = -(recon_loss + self.beta * kl)

        return GaussianELBOOutput(
            elbo=elbo.mean(),
            recon_loss=recon_loss.mean(),
            kl=kl.mean(),
            recon=recon,
            mu=mu,
            log_sigma=log_sigma,
        )

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, GaussianELBOOutput]:
        out = self.forward(x)
        return -out.elbo, out
