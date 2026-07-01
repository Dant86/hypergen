"""TNBbeta p × q ablation grid: sphere heatmaps with scalar marginals below.

Usage:
    uv run python apps/plot_pq_ablation.py
    uv run python apps/plot_pq_ablation.py --output plots/pq_ablation.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch

from hypergen.distributions.tnbbeta import TNBbeta

MU = np.array([0.0, 0.0, 1.0])
EPS = 1.0
P_VALS = [0.05, 0.2, 0.5, 0.8, 0.95]
Q_VALS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def sphere_mesh(n: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, n)
    phi, theta = np.meshgrid(phi, theta)
    return np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)


def eval_sphere_log_density(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    p: float, q: float,
) -> np.ndarray:
    cos_sim = x * MU[0] + y * MU[1] + z * MU[2]
    r = np.clip((1.0 + cos_sim) / 2.0, 1e-6, 1.0 - 1e-6)
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(EPS, dtype=torch.float64),
    )
    with torch.no_grad():
        return dist.log_prob(torch.tensor(r, dtype=torch.float64)).numpy()


def eval_scalar_density(y_grid: np.ndarray, p: float, q: float) -> np.ndarray:
    dist = TNBbeta(
        p=torch.tensor(p, dtype=torch.float64),
        q=torch.tensor(q, dtype=torch.float64),
        eps=torch.tensor(EPS, dtype=torch.float64),
    )
    with torch.no_grad():
        return torch.exp(dist.log_prob(torch.tensor(y_grid, dtype=torch.float64))).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("plots/pq_ablation.png"))
    parser.add_argument("--resolution", type=int, default=150)
    args = parser.parse_args()

    n_rows = len(P_VALS)
    n_cols = len(Q_VALS)
    sx, sy, sz = sphere_mesh(args.resolution)
    y_grid = np.linspace(0.005, 0.995, 400)
    cmap = cm.inferno  # type: ignore[attr-defined]

    sphere_w = 0.12
    sphere_h = 0.11
    curve_h = 0.04
    gap_y = 0.005
    left_margin = 0.07
    top_margin = 0.06
    col_gap = 0.005
    row_gap = 0.015

    total_w = left_margin + n_cols * (sphere_w + col_gap)
    total_h = top_margin + n_rows * (sphere_h + curve_h + gap_y + row_gap)
    fig = plt.figure(figsize=(total_w * 80, total_h * 55))

    for ri, p_val in enumerate(P_VALS):
        for ci, q_val in enumerate(Q_VALS):
            x0 = left_margin + ci * (sphere_w + col_gap)
            y0 = 1.0 - top_margin - (ri + 1) * (sphere_h + curve_h + gap_y + row_gap) + row_gap

            # Sphere
            ax3d = fig.add_axes(
                [x0, y0 + curve_h + gap_y, sphere_w, sphere_h], projection="3d"
            )
            density = eval_sphere_log_density(sx, sy, sz, p_val, q_val)
            norm = Normalize(vmin=density.min(), vmax=density.max())
            colors = cmap(norm(density))
            ax3d.plot_surface(  # type: ignore[attr-defined]
                sx, sy, sz, facecolors=colors, rstride=2, cstride=2,
                antialiased=False, shade=False,
            )
            ax3d.set_xlim([-1.1, 1.1])
            ax3d.set_ylim([-1.1, 1.1])
            ax3d.set_zlim([-1.1, 1.1])  # type: ignore[attr-defined]
            ax3d.set_box_aspect([1, 1, 1])  # type: ignore[attr-defined]
            ax3d.view_init(elev=50, azim=45)  # type: ignore[attr-defined]
            ax3d.set_proj_type("ortho")  # type: ignore[attr-defined]
            ax3d.set_axis_off()

            # Column header
            if ri == 0:
                ax3d.set_title(f"$q={q_val}$", fontsize=9, pad=2)

            # Row label
            if ci == 0:
                ax3d.text2D(  # type: ignore[attr-defined]
                    -0.15, 0.5, f"$p={p_val}$",
                    transform=ax3d.transAxes, fontsize=9,
                    ha="center", va="center", rotation=90,
                )

            # Scalar density curve
            ax_curve = fig.add_axes([x0 + 0.01, y0, sphere_w - 0.02, curve_h])
            dens = eval_scalar_density(y_grid, p_val, q_val)
            ax_curve.fill_between(y_grid, dens, alpha=0.3, color="#e7298a")
            ax_curve.plot(y_grid, dens, color="#e7298a", lw=1.2)
            ax_curve.set_xlim(0, 1)
            ax_curve.set_ylim(bottom=0)
            ax_curve.set_xticks([])
            ax_curve.set_yticks([])
            ax_curve.spines["top"].set_visible(False)
            ax_curve.spines["right"].set_visible(False)
            ax_curve.spines["left"].set_visible(False)

            # Only show x-axis labels on bottom row
            if ri == n_rows - 1:
                ax_curve.set_xticks([0, 0.5, 1])
                ax_curve.tick_params(labelsize=6)
            else:
                ax_curve.spines["bottom"].set_linewidth(0.5)

    fig.suptitle(
        r"TNBbeta $p \times q$ ablation on $S^2$ $(\varepsilon=1.0,\;\mu=[0,0,1])$",
        fontsize=12, y=0.98,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
