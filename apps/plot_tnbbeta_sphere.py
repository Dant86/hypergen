"""3D sphere heatmaps of the TNBbeta spherical distribution for varying parameters.

Renders a 2x2 grid of S^2 density plots, each with different (p, q, eps, mu).
The density is rotationally symmetric around mu, so we evaluate the scalar
TNBbeta density at the alignment coordinate R = (1 + mu^T x) / 2 for each
point on the sphere mesh and color by log-density.

Usage:
    uv run python apps/plot_tnbbeta_sphere.py
    uv run python apps/plot_tnbbeta_sphere.py --output plots/tnbbeta_sphere.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize

from hypergen.distributions.tnbbeta import TNBbeta


def sphere_mesh(n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def eval_density(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mu: np.ndarray,
    p: float,
    q: float,
    eps: float,
) -> np.ndarray:
    cos_sim = x * mu[0] + y * mu[1] + z * mu[2]
    r = np.clip((1.0 + cos_sim) / 2.0, 1e-6, 1.0 - 1e-6)
    r_t = torch.tensor(r, dtype=torch.float64)
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(eps, dtype=torch.float64),
    )
    with torch.no_grad():
        log_p = dist.log_prob(r_t)
    return log_p.numpy()


CONFIGS = [
    {
        "title": r"$p=0.5,\; q=0.0,\; \varepsilon=3.0$"
        + "\n(symmetric, moderate concentration)",
        "p": 0.5,
        "q": 0.0,
        "eps": 3.0,
        "mu": np.array([0.0, 0.0, 1.0]),
    },
    {
        "title": r"$p=0.5,\; q=0.0,\; \varepsilon=10.0$"
        + "\n(symmetric, high concentration)",
        "p": 0.5,
        "q": 0.0,
        "eps": 10.0,
        "mu": np.array([0.0, 0.0, 1.0]),
    },
    {
        "title": r"$p=0.75,\; q=0.2,\; \varepsilon=5.0$"
        + "\n(off-centre mode, heavy-tailed)",
        "p": 0.75,
        "q": 0.2,
        "eps": 5.0,
        "mu": np.array([0.0, 0.0, 1.0]),
    },
    {
        "title": r"$p=0.7,\; q=0.15,\; \varepsilon=6.0,\; \mu=\frac{1}{\sqrt{3}}[1,1,1]$"
        + "\n(tilted mode, moderate concentration)",
        "p": 0.7,
        "q": 0.15,
        "eps": 6.0,
        "mu": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("plots/tnbbeta_sphere.png"))
    parser.add_argument("--resolution", type=int, default=200)
    args = parser.parse_args()

    x, y, z = sphere_mesh(args.resolution)

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 14),
        subplot_kw={"projection": "3d"},
    )
    cmap = cm.inferno  # type: ignore[attr-defined]

    for idx, cfg in enumerate(CONFIGS):
        ax = axes.flat[idx]
        density = eval_density(x, y, z, cfg["mu"], cfg["p"], cfg["q"], cfg["eps"])

        norm = Normalize(vmin=density.min(), vmax=density.max())
        colors = cmap(norm(density))
        ax.plot_surface(  # type: ignore[attr-defined]
            x, y, z,
            facecolors=colors,
            rstride=1, cstride=1,
            antialiased=False, shade=False,
        )

        mu = cfg["mu"]
        ax.quiver(  # type: ignore[attr-defined]
            0, 0, 0, mu[0] * 1.5, mu[1] * 1.5, mu[2] * 1.5,
            color="cyan", arrow_length_ratio=0.08, linewidth=2.0, alpha=0.9,
        )

        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])  # type: ignore[attr-defined]
        ax.set_box_aspect([1, 1, 1])  # type: ignore[attr-defined]
        ax.view_init(elev=25, azim=45)  # type: ignore[attr-defined]
        ax.set_axis_off()
        ax.set_title(cfg["title"], fontsize=10, pad=-5)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, aspect=18, pad=-0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("log density", fontsize=9)

    fig.suptitle(
        "TNBbeta Spherical Distribution on $S^2$",
        fontsize=14,
        x=0.45,
        y=0.99,
    )
    fig.subplots_adjust(left=0.02, right=0.88, top=0.93, bottom=0.02, wspace=-0.15, hspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
