"""Baseline VAEs sharing the encoder/decoder backbone, swapping the latent distribution."""

from hypergen.baselines.gaussian_vae import GaussianVAE
from hypergen.baselines.power_spherical_vae import PowerSphericalVAE
from hypergen.baselines.vmf_vae import VMFVAE

__all__ = ["GaussianVAE", "PowerSphericalVAE", "VMFVAE"]
