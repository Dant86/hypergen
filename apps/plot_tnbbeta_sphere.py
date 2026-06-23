"""Hybrid visualization of the TNBbeta distribution.

Top row: three panels of scalar TNBbeta density on (0,1), each varying one
parameter (p, eps, q) while holding the others fixed.
Bottom row: a single 3D sphere heatmap showing the lifted density on S^2.

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


def eval_scalar_density(y: np.ndarray, p: float, q: float, eps: float) -> np.ndarray:
    y_t = torch.tensor(y, dtype=torch.float64)
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(eps, dtype=torch.float64),
    )
    with torch.no_grad():
        return torch.exp(dist.log_prob(y_t)).numpy()


def eval_sphere_density(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    mu: np.ndarray, p: float, q: float, eps: float,
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
        return dist.log_prob(r_t).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("plots/tnbbeta_sphere.png"))
    parser.add_argument("--resolution", type=int, default=200)
    args = parser.parse_args()

    y_grid = np.linspace(0.005, 0.995, 500)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], hspace=0.32, wspace=0.3,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── Top row: 1D scalar densities ─────────────────────────────────────────

    # Panel 1: vary p (location/mode)
    ax0 = fig.add_subplot(gs[0, 0])
    base_eps, base_q = 5.0, 0.0
    for p_val, color in [(0.3, "#1b9e77"), (0.5, "#d95f02"), (0.7, "#7570b3"), (0.9, "#e7298a")]:
        density = eval_scalar_density(y_grid, p_val, base_q, base_eps)
        ax0.plot(y_grid, density, color=color, lw=2, label=f"$p={p_val}$")
    ax0.set_xlabel("$y$", fontsize=11)
    ax0.set_ylabel("density", fontsize=11)
    ax0.set_title(r"Varying $p$ (mode location)" + f"\n$q={base_q},\\; \\varepsilon={base_eps}$",
                  fontsize=10)
    ax0.legend(fontsize=9)
    ax0.set_xlim(0, 1)

    # Panel 2: vary eps (concentration)
    ax1 = fig.add_subplot(gs[0, 1])
    base_p, base_q = 0.7, 0.0
    for eps_val, color in [(1.0, "#1b9e77"), (3.0, "#d95f02"), (8.0, "#7570b3"), (15.0, "#e7298a")]:
        density = eval_scalar_density(y_grid, base_p, base_q, eps_val)
        ax1.plot(y_grid, density, color=color, lw=2, label=f"$\\varepsilon={eps_val}$")
    ax1.set_xlabel("$y$", fontsize=11)
    ax1.set_title(r"Varying $\varepsilon$ (concentration)" + f"\n$p={base_p},\\; q={base_q}$",
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)

    # Panel 3: vary q (tail weight)
    ax2 = fig.add_subplot(gs[0, 2])
    base_p, base_eps = 0.7, 5.0
    for q_val, color in [(0.0, "#1b9e77"), (0.2, "#d95f02"), (0.5, "#7570b3"), (0.9, "#e7298a")]:
        density = eval_scalar_density(y_grid, base_p, q_val, base_eps)
        ax2.plot(y_grid, density, color=color, lw=2, label=f"$q={q_val}$")
    ax2.set_xlabel("$y$", fontsize=11)
    ax2.set_title(r"Varying $q$ (tail weight)" + f"\n$p={base_p},\\; \\varepsilon={base_eps}$",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)

    # ── Bottom row: single sphere ────────────────────────────────────────────

    ax3d = fig.add_subplot(gs[1, :], projection="3d")
    sx, sy, sz = sphere_mesh(args.resolution)
    mu = np.array([0.0, 0.0, 1.0])
    sphere_p, sphere_q, sphere_eps = 0.85, 0.1, 6.0
    density = eval_sphere_density(sx, sy, sz, mu, sphere_p, sphere_q, sphere_eps)

    cmap = cm.inferno  # type: ignore[attr-defined]
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colors = cmap(norm(density))
    ax3d.plot_surface(  # type: ignore[attr-defined]
        sx, sy, sz,
        facecolors=colors, rstride=2, cstride=2,
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
        0, 0, 0, mu[0] * 1.5, mu[1] * 1.5, mu[2] * 1.5,
        color="cyan", arrow_length_ratio=0.06, linewidth=2.0, alpha=0.9,
    )
    arrow_len = 1.5
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
            label, fontsize=9, color="black", ha="center", va="center",
        )

    ax3d.set_xlim([-1.3, 1.3])
    ax3d.set_ylim([-1.3, 1.3])
    ax3d.set_zlim([-1.3, 1.3])  # type: ignore[attr-defined]
    ax3d.set_box_aspect([1, 1, 1])  # type: ignore[attr-defined]
    ax3d.view_init(elev=50, azim=45)  # type: ignore[attr-defined]
    ax3d.set_axis_off()
    ax3d.set_title(
        f"Lifted to $S^2$:  $p={sphere_p},\\; q={sphere_q},"
        f"\\; \\varepsilon={sphere_eps},\\; \\mu=[0,0,1]$",
        fontsize=11, pad=-5,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.5, aspect=20, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("log density", fontsize=9)

    fig.suptitle(
        "TNBbeta Distribution: Scalar Density on $(0,1)$ and Lifted to $S^2$",
        fontsize=14, y=0.97,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
