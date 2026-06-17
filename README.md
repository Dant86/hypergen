# hypergen

Hyperspherical Latent Generative Models.

Implements a Variational Autoencoder using the **TNBbeta** distribution on the
unit interval, lifted to the unit hypersphere `S^{d-1}` via a Householder
transformation, trained on CIFAR-100 and benchmarked against vMF (S-VAE),
Power Spherical, and Gaussian VAE baselines.

## Layout

- `src/hypergen/distributions/` — `TNBbeta` (base distribution) and
  `TNBbetaSpherical` (Householder lift to `S^{d-1}`).
- `src/hypergen/models/` — encoder, decoder, and the full `TNBbetaVAE`.
- `src/hypergen/baselines/` — `GaussianVAE`, `VMFVAE`, `PowerSphericalVAE`.
- `src/hypergen/metrics/` — isotropy score and linear-probe utilities.
- `apps/` — training (`train_cifar100.py`), ablations (`eval_latent.py`),
  and plotting (`plot_results.py`) scripts.
- `tests/` — pytest suite covering the distribution, the lift, the VAE, and
  the metrics.

## Setup

```bash
uv sync --all-extras
```

## Usage

```bash
uv run python apps/train_cifar100.py --model tnbbeta --latent-dim 64 --beta 1.0
uv run python apps/eval_latent.py --model tnbbeta --checkpoint checkpoints/tnbbeta_epoch200.pt --ablation isotropy
uv run python apps/plot_results.py --model tnbbeta --checkpoint checkpoints/tnbbeta_epoch200.pt
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest tests/ -v
```

## References

- Lederman & Schein (2026) TNBbeta: https://arxiv.org/abs/2606.11624
- Davidson et al. (2018) S-VAE: https://arxiv.org/abs/1804.00891
- Davidson et al. (2019) Increasing Expressivity: https://arxiv.org/abs/1910.02912
- De Cao & Aziz (2020) Power Spherical: https://arxiv.org/abs/2006.04437
- Sablica & Hornik (2025) spCauchy VAE: https://arxiv.org/abs/2506.21278
