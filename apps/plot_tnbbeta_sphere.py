"""TNBbeta distribution visualization: three columns, each with a sphere
on top and the corresponding scalar density curve beneath.

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
    return np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)


def eval_scalar_density(y: np.ndarray, p: float, q: float, eps: float) -> np.ndarray:
    y_t = torch.tensor(y, dtype=torch.float64)
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(eps, dtype=torch.float64),
    )
    with torch.no_grad():
        return torch.exp(dist.log_prob(y_t)).numpy()


def eval_sphere_log_density(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    mu: np.ndarray, p: float, q: float, eps: float,
) -> np.ndarray:
    cos_sim = x * mu[0] + y * mu[1] + z * mu[2]
    r = np.clip((1.0 + cos_sim) / 2.0, 1e-6, 1.0 - 1e-6)
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(eps, dtype=torch.float64),
    )
    with torch.no_grad():
        return dist.log_prob(torch.tensor(r, dtype=torch.float64)).numpy()


MU = np.array([0.0, 0.0, 1.0])

CONFIGS = [
    {"label": "Low concentration\n$p=0.8,\\; q=0.1,\\; \\varepsilon=3.0$",
     "p": 0.8, "q": 0.1, "eps": 3.0},
    {"label": "High concentration\n$p=0.8,\\; q=0.9,\\; \\varepsilon=3.0$",
     "p": 0.8, "q": 0.9, "eps": 3.0},
    {"label": "Heavy tails\n$p=0.8,\\; q=0.5,\\; \\varepsilon=0.4$",
     "p": 0.8, "q": 0.5, "eps": 0.4},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("plots/tnbbeta_sphere.png"))
    parser.add_argument("--resolution", type=int, default=200)
    args = parser.parse_args()

    sx, sy, sz = sphere_mesh(args.resolution)
    y_grid = np.linspace(0.005, 0.995, 500)
    cmap = cm.inferno  # type: ignore[attr-defined]

    fig = plt.figure(figsize=(15, 6.5))

    col_centers = [0.18, 0.48, 0.78]
    sphere_w, sphere_h = 0.28, 0.58
    curve_w, curve_h = 0.22, 0.18

    for i, cfg in enumerate(CONFIGS):
        cx = col_centers[i]

        # Sphere (top)
        ax3d = fig.add_axes([cx - sphere_w / 2, 0.30, sphere_w, sphere_h],
                            projection="3d")
        density = eval_sphere_log_density(sx, sy, sz, MU, cfg["p"], cfg["q"], cfg["eps"])
        norm = Normalize(vmin=density.min(), vmax=density.max())
        colors = cmap(norm(density))
        ax3d.plot_surface(  # type: ignore[attr-defined]
            sx, sy, sz, facecolors=colors, rstride=2, cstride=2,
            antialiased=False, shade=False,
        )
        r_wire = 1.005
        n_wire = 100
        for lat in np.linspace(0, np.pi, 7)[1:-1]:
            th = np.linspace(0, 2 * np.pi, n_wire)
            ax3d.plot(  # type: ignore[attr-defined]
                r_wire * np.sin(lat) * np.cos(th),
                r_wire * np.sin(lat) * np.sin(th),
                r_wire * np.cos(lat) * np.ones_like(th),
                color="black", lw=0.3, alpha=0.35,
            )
        for lon in np.linspace(0, 2 * np.pi, 13)[:-1]:
            ph = np.linspace(0, np.pi, n_wire)
            ax3d.plot(  # type: ignore[attr-defined]
                r_wire * np.sin(ph) * np.cos(lon),
                r_wire * np.sin(ph) * np.sin(lon),
                r_wire * np.cos(ph),
                color="black", lw=0.3, alpha=0.35,
            )
        ax3d.quiver(  # type: ignore[attr-defined]
            0, 0, 0, MU[0] * 1.4, MU[1] * 1.4, MU[2] * 1.4,
            color="cyan", arrow_length_ratio=0.06, linewidth=2.0, alpha=0.9,
        )
        ax3d.set_xlim([-1.1, 1.1])
        ax3d.set_ylim([-1.1, 1.1])
        ax3d.set_zlim([-1.1, 1.1])  # type: ignore[attr-defined]
        ax3d.set_box_aspect([1, 1, 1])  # type: ignore[attr-defined]
        ax3d.view_init(elev=50, azim=45)  # type: ignore[attr-defined]
        ax3d.set_proj_type("ortho")  # type: ignore[attr-defined]
        arrow_len = 1.4
        for vec, label in [
            ([arrow_len, 0, 0], "$\\hat{x}$"),
            ([0, arrow_len, 0], "$\\hat{y}$"),
            ([0, 0, arrow_len], "$\\hat{z}$"),
        ]:
            ax3d.quiver(  # type: ignore[attr-defined]
                0, 0, 0, vec[0], vec[1], vec[2],
                color="black", arrow_length_ratio=0.05, linewidth=1.0, alpha=0.7,
            )
            ax3d.text(  # type: ignore[attr-defined]
                vec[0] * 1.12, vec[1] * 1.12, vec[2] * 1.12,
                label, fontsize=8, color="black", ha="center", va="center",
            )
        ax3d.set_axis_off()
        ax3d.set_title(cfg["label"], fontsize=10, y=0.95)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax3d, shrink=0.55, aspect=14, pad=-0.01)
        cbar.ax.tick_params(labelsize=7)
        if i == len(CONFIGS) - 1:
            cbar.set_label("log density", fontsize=8)

        # Density curve (bottom)
        ax_curve = fig.add_axes([cx - curve_w / 2, 0.14, curve_w, curve_h])
        dens = eval_scalar_density(y_grid, cfg["p"], cfg["q"], cfg["eps"])
        ax_curve.fill_between(y_grid, dens, alpha=0.3, color="#e7298a")
        ax_curve.plot(y_grid, dens, color="#e7298a", lw=2)
        ax_curve.set_xlabel("$y$", fontsize=10)
        if i == 0:
            ax_curve.set_ylabel("density", fontsize=10)
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(bottom=0)

    fig.suptitle(
        "TNBbeta Distribution on $(0,1)$ and Lifted to $S^2$"
        "$\\quad(\\mu = [0,0,1])$",
        fontsize=13, y=0.99,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
