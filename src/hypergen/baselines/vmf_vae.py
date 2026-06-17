"""von Mises-Fisher VAE baseline (Davidson et al., 2018), sharing the backbone.

Reimplements vMF sampling via the rejection-sampling scheme of Ulrich (1984) /
Wood (1994), as used in the S-VAE paper, with the reparameterization trick
applied to the auxiliary uniform variable on S^{d-2}. The modified Bessel
function I_nu(kappa) (needed for the normalizer/entropy) has no native
PyTorch op for arbitrary real order, so it is implemented as a custom
autograd.Function backed by `scipy.special.ive`, using the standard
recurrence relation for the gradient -- the same trick used by the
`hyperspherical-vae` reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from power_spherical import HypersphericalUniform
from scipy import special
import torch
from torch import nn
from torch.autograd import Function
from torch.distributions import Beta, Distribution

from hypergen.distributions.spherical import householder_matrix
from hypergen.models.decoder import ConvDecoder
from hypergen.models.encoder import ConvBackbone

_EPS_CLAMP = 1e-7


class _LogModifiedBesselFn(Function):
    """log I_nu(kappa), with gradient via the recurrence I_nu' = I_{nu-1} - (nu/kappa) I_nu."""

    @staticmethod
    def forward(ctx: object, nu: float, kappa: torch.Tensor) -> torch.Tensor:
        kappa_np = kappa.detach().cpu().numpy()
        log_iv = (
            torch.as_tensor(special.ive(nu, kappa_np), dtype=kappa.dtype, device=kappa.device).log()
            + kappa
        )
        ctx.save_for_backward(kappa)  # type: ignore[attr-defined]
        ctx.nu = nu  # type: ignore[attr-defined]
        return log_iv

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        (kappa,) = ctx.saved_tensors  # type: ignore[attr-defined]
        nu: float = ctx.nu  # type: ignore[attr-defined]
        d_log_iv = log_iv(nu - 1.0, kappa).exp() / log_iv(nu, kappa).exp() - nu / kappa.clamp_min(
            _EPS_CLAMP
        )
        return None, grad_output * d_log_iv


def log_iv(nu: float, kappa: torch.Tensor) -> torch.Tensor:
    """Differentiable log of the modified Bessel function of the first kind."""
    return _LogModifiedBesselFn.apply(nu, kappa)  # type: ignore[no-any-return]


class VonMisesFisher(Distribution):
    """von Mises-Fisher distribution on S^{d-1} via Ulrich/Wood rejection sampling."""

    has_rsample = True

    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor) -> None:
        self.mu = mu
        self.kappa = kappa
        self.dim = mu.shape[-1]
        super().__init__(batch_shape=mu.shape[:-1], validate_args=False)

    def _sample_w(self, shape: torch.Size) -> torch.Tensor:
        d = self.dim
        kappa = self.kappa.expand(shape + self.kappa.shape)
        b = (-2.0 * kappa + torch.sqrt(4.0 * kappa.pow(2) + (d - 1.0) ** 2)) / (d - 1.0)
        eps_b = Beta((d - 1.0) / 2.0, (d - 1.0) / 2.0).rsample()
        return (1.0 - (1.0 + b) * eps_b) / (1.0 - (1.0 - b) * eps_b)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        shape = torch.Size(sample_shape)
        w = self._sample_w(shape).clamp(-1.0 + _EPS_CLAMP, 1.0 - _EPS_CLAMP)

        v = torch.randn(
            *shape, *self.mu.shape[:-1], self.dim - 1, dtype=self.mu.dtype, device=self.mu.device
        )
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(_EPS_CLAMP)

        ortho_scale = torch.sqrt((1.0 - w.pow(2)).clamp_min(0.0)).unsqueeze(-1)
        x_north = torch.cat([w.unsqueeze(-1), ortho_scale * v], dim=-1)

        mu_expanded = self.mu.expand(shape + self.mu.shape)
        house = householder_matrix(mu_expanded)
        x = torch.einsum("...ij,...j->...i", house, x_north)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(_EPS_CLAMP)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:  # noqa: B008
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_normalizer(self) -> torch.Tensor:
        d = self.dim
        kappa = self.kappa.clamp_min(_EPS_CLAMP)
        nu = d / 2.0 - 1.0
        return (
            (d / 2.0 - 1.0) * torch.log(kappa)
            - (d / 2.0) * math.log(2.0 * math.pi)
            - log_iv(nu, kappa)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        cos_sim = (self.mu * value).sum(dim=-1)
        return self.log_normalizer() + self.kappa * cos_sim

    def entropy(self) -> torch.Tensor:
        d = self.dim
        kappa = self.kappa.clamp_min(_EPS_CLAMP)
        nu = d / 2.0
        ratio = log_iv(nu, kappa).exp() / log_iv(nu - 1.0, kappa).exp()
        return -self.log_normalizer() - kappa * ratio


@dataclass
class VMFELBOOutput:
    """Container for the components of the vMF VAE ELBO."""

    elbo: torch.Tensor
    recon_loss: torch.Tensor
    kl: torch.Tensor
    recon: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor


class VMFVAE(nn.Module):
    """VAE with a vMF(mu, kappa) posterior and a Uniform(S^{d-1}) prior."""

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

    def forward(self, x: torch.Tensor) -> VMFELBOOutput:
        mu, kappa = self.encode(x)
        posterior = VonMisesFisher(mu, kappa)

        z = posterior.rsample()
        recon = self.decoder(z)
        recon_loss = (
            nn.functional.binary_cross_entropy_with_logits(recon, x, reduction="none")
            .flatten(1)
            .sum(dim=1)
        )

        uniform = HypersphericalUniform(self.latent_dim, device=str(mu.device), dtype=mu.dtype)
        log_p_uniform = uniform.log_prob(z)
        kl = (-posterior.entropy() - log_p_uniform).clamp_min(0.0)
        elbo = -(recon_loss + self.beta * kl)

        return VMFELBOOutput(
            elbo=elbo.mean(),
            recon_loss=recon_loss.mean(),
            kl=kl.mean(),
            recon=recon,
            mu=mu,
            kappa=kappa,
        )

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, VMFELBOOutput]:
        out = self.forward(x)
        return -out.elbo, out
