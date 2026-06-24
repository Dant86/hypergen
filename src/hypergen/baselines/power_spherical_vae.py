"""Power Spherical VAE baseline (De Cao & Aziz, 2020), sharing the backbone."""

from __future__ import annotations

from dataclasses import dataclass

from power_spherical import HypersphericalUniform, PowerSpherical
import torch
from torch import nn
from torch.distributions import kl_divergence

from hypergen.models.decoder import ConvDecoder
from hypergen.models.encoder import ConvBackbone


@dataclass
class PowerSphericalELBOOutput:
    """Container for the components of the Power Spherical VAE ELBO."""

    elbo: torch.Tensor
    recon_loss: torch.Tensor
    kl: torch.Tensor
    recon: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor


class PowerSphericalVAE(nn.Module):
    """VAE with a PowerSpherical(mu, kappa) posterior and a Uniform(S^{d-1}) prior."""

    def __init__(self, latent_dim: int = 64, beta: float = 1.0, feature_dim: int = 512) -> None:
        super().__init__()
        self.backbone = ConvBackbone(feature_dim=feature_dim)
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.kappa_head = nn.Linear(feature_dim, 1)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        mu_raw = self.mu_head(feats)
        mu = mu_raw / mu_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        kappa = nn.functional.softplus(self.kappa_head(feats).squeeze(-1)) + 1e-3
        return mu, kappa

    def forward(self, x: torch.Tensor) -> PowerSphericalELBOOutput:
        mu, kappa = self.encode(x)
        posterior = PowerSpherical(mu, kappa)
        prior = HypersphericalUniform(self.latent_dim, device=str(mu.device), dtype=mu.dtype)

        z = posterior.rsample()
        recon = self.decoder(z)
        recon_loss = (
            nn.functional.mse_loss(torch.sigmoid(recon), x, reduction="none")
            .flatten(1)
            .sum(dim=1)
        )

        kl = kl_divergence(posterior, prior)
        elbo = -(recon_loss + self.beta * kl)

        return PowerSphericalELBOOutput(
            elbo=elbo.mean(),
            recon_loss=recon_loss.mean(),
            kl=kl.mean(),
            recon=recon,
            mu=mu,
            kappa=kappa,
        )

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, PowerSphericalELBOOutput]:
        out = self.forward(x)
        return -out.elbo, out
