"""Full TNBbeta-VAE: encoder + decoder + ELBO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Beta

from hypergen.distributions.spherical import TNBbetaSpherical
from hypergen.models.decoder import ConvDecoder
from hypergen.models.encoder import TNBbetaEncoder


@dataclass
class ELBOOutput:
    """Container for the components of the ELBO."""

    elbo: torch.Tensor
    recon_loss: torch.Tensor
    kl: torch.Tensor
    recon: torch.Tensor
    mu: torch.Tensor
    p: torch.Tensor
    q: torch.Tensor
    eps: torch.Tensor


class TNBbetaVAE(nn.Module):
    """VAE with a TNBbeta-Spherical posterior and a Uniform(S^{d-1}) prior.

    Args:
        latent_dim: Dimensionality d of the latent sphere.
        beta: Coefficient on the KL term (beta-VAE).
        kl_mc_samples: Number of Monte Carlo samples for the KL estimate.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        beta: float = 1.0,
        kl_mc_samples: int = 64,
        fixed_eps: float | None = 1.0,
        q_max: float = 0.25,
        w1_beta: float = 500.0,
    ) -> None:
        super().__init__()
        self.encoder = TNBbetaEncoder(latent_dim=latent_dim, fixed_eps=fixed_eps, q_max=q_max)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta
        self.w1_beta = w1_beta
        self.kl_mc_samples = kl_mc_samples

    def regularizer(self, posterior: TNBbetaSpherical) -> torch.Tensor:
        """W_1(TNBbeta(p,q,eps) || Beta((d-1)/2, (d-1)/2)) on the scalar marginal."""
        alpha = torch.tensor(
            (self.latent_dim - 1) / 2.0,
            dtype=posterior.mu.dtype,
            device=posterior.mu.device,
        )
        prior = Beta(alpha, alpha)
        return posterior.tnbbeta.wasserstein1(prior, n_samples=self.kl_mc_samples)

    def forward(self, x: torch.Tensor) -> ELBOOutput:
        mu, p, q, eps = self.encoder(x)
        posterior = TNBbetaSpherical(mu, p, q, eps)
        z = posterior.rsample()

        recon = self.decoder(z)
        recon_loss = (
            nn.functional.mse_loss(torch.sigmoid(recon), x, reduction="none")
            .flatten(1)
            .sum(dim=1)
        )

        w1 = self.regularizer(posterior)
        elbo = -(recon_loss + self.w1_beta * w1)

        return ELBOOutput(
            elbo=elbo.mean(),
            recon_loss=recon_loss.mean(),
            kl=w1.mean(),
            recon=recon,
            mu=mu,
            p=p,
            q=q,
            eps=eps,
        )

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, ELBOOutput]:
        out = self.forward(x)
        return -out.elbo, out
