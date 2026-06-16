"""hypergen: Hyperspherical Latent Generative Models.

TNBbeta-VAE and baselines (Gaussian, vMF, Power Spherical).
"""

from hypergen.distributions import TNBbeta, TNBbetaSpherical
from hypergen.models import ConvDecoder, TNBbetaEncoder, TNBbetaVAE

__all__ = [
    "ConvDecoder",
    "TNBbeta",
    "TNBbetaEncoder",
    "TNBbetaSpherical",
    "TNBbetaVAE",
]
