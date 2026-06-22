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

    def __init__(self, latent_dim: int = 64, beta: float = 1.0, kl_mc_samples: int = 64) -> None:
        super().__init__()
        self.encoder = TNBbetaEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta
        self.kl_mc_samples = kl_mc_samples

    def kl_divergence(self, posterior: TNBbetaSpherical) -> torch.Tensor:
        """MC estimate of KL(TNBbeta(p,q,eps) || Beta((d-1)/2, (d-1)/2)) on the scalar marginal.

        The alignment coordinate R = (1+mu^T x)/2 of a uniform point on
        S^{d-1} follows Beta((d-1)/2, (d-1)/2), so that is the correct
        scalar prior.
        """
        alpha = torch.tensor((self.latent_dim - 1) / 2.0, device=posterior.mu.device)
        prior = Beta(alpha, alpha)
        samples = posterior.tnbbeta.rsample(torch.Size([self.kl_mc_samples]))
        log_q = posterior.tnbbeta.log_prob(samples)
        log_p = prior.log_prob(samples)
        return (log_q - log_p).mean(dim=0)

    def forward(self, x: torch.Tensor) -> ELBOOutput:
        mu, p, q, eps = self.encoder(x)
        posterior = TNBbetaSpherical(mu, p, q, eps)
        z = posterior.rsample()

        recon = self.decoder(z)
        recon_loss = (
            nn.functional.binary_cross_entropy_with_logits(recon, x, reduction="none")
            .flatten(1)
            .sum(dim=1)
        )

        kl = self.kl_divergence(posterior)
        elbo = -(recon_loss + self.beta * kl)

        return ELBOOutput(
            elbo=elbo.mean(),
            recon_loss=recon_loss.mean(),
            kl=kl.mean(),
            recon=recon,
            mu=mu,
            p=p,
            q=q,
            eps=eps,
        )

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, ELBOOutput]:
        out = self.forward(x)
        return -out.elbo, out
