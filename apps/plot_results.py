"""Visualization helpers for hypergen ablations: SLERP grids and sweep plots.

Usage:
    uv run python apps/plot_results.py --checkpoint checkpoints/tnbbeta_epoch200.pt --model tnbbeta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from hypergen.baselines import VMFVAE, GaussianVAE, PowerSphericalVAE
from hypergen.models import TNBbetaVAE

MODEL_REGISTRY = {
    "gaussian": GaussianVAE,
    "vmf": VMFVAE,
    "power_spherical": PowerSphericalVAE,
    "tnbbeta": TNBbetaVAE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SLERP interpolations and sweep results.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="tnbbeta")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("slerp_grid.png"))
    return parser.parse_args()


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between unit vectors z0 and z1."""
    dot = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-7)
    t = t.unsqueeze(-1)
    return (torch.sin((1.0 - t) * omega) * z0 + torch.sin(t * omega) * z1) / sin_omega


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = torchvision.datasets.CIFAR100(
        root=str(args.data_dir), train=False, download=True, transform=transforms.ToTensor()
    )
    idx0: list[int] = torch.randint(0, len(dataset), (args.num_pairs,)).tolist()
    idx1: list[int] = torch.randint(0, len(dataset), (args.num_pairs,)).tolist()
    images0 = torch.stack([dataset[i][0] for i in idx0]).to(device)
    images1 = torch.stack([dataset[i][0] for i in idx1]).to(device)

    with torch.no_grad():
        z0 = model.forward(images0).mu
        z1 = model.forward(images1).mu

        t = torch.linspace(0.0, 1.0, args.num_steps, device=device)
        decoded_rows = []
        for pair_idx in range(args.num_pairs):
            zs = slerp(
                z0[pair_idx].expand(args.num_steps, -1), z1[pair_idx].expand(args.num_steps, -1), t
            )
            decoded_rows.append(torch.sigmoid(model.decoder(zs)))
        grid_images = torch.cat(decoded_rows, dim=0)

    grid = make_grid(grid_images, nrow=args.num_steps)
    plt.figure(figsize=(args.num_steps, args.num_pairs))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.savefig(args.output, bbox_inches="tight")
    print(f"saved SLERP grid to {args.output}")


if __name__ == "__main__":
    main()
