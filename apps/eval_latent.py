"""Latent-space ablations for hypergen VAEs.

Usage:
    uv run python apps/eval_latent.py --checkpoint checkpoints/tnbbeta_epoch200.pt --model tnbbeta
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from hypergen.baselines import VMFVAE, GaussianVAE, PowerSphericalVAE
from hypergen.metrics import LinearProbe, active_units, isotropy_score
from hypergen.models import TNBbetaVAE

MODEL_REGISTRY = {
    "gaussian": GaussianVAE,
    "vmf": VMFVAE,
    "power_spherical": PowerSphericalVAE,
    "tnbbeta": TNBbetaVAE,
}

P_SWEEP = [0.2, 0.35, 0.5, 0.65, 0.8]
Q_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2]
EPS_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0]
LATENT_DIM_SWEEP = [16, 32, 64, 128, 256]

CIFAR100Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-space ablations.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="tnbbeta")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--ablation",
        choices=["param_sweep", "dim_scaling", "isotropy", "probe"],
        default="isotropy",
    )
    return parser.parse_args()


def build_eval_loader(data_dir: Path, batch_size: int) -> CIFAR100Loader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def collect_mus(
    model: torch.nn.Module,
    loader: CIFAR100Loader,
    device: torch.device,
) -> torch.Tensor:
    all_mus = []
    for images, _labels in loader:
        images = images.to(device)
        out = model.forward(images)
        all_mus.append(out.mu.cpu())
    return torch.cat(all_mus, dim=0)


def run_isotropy(
    model: torch.nn.Module,
    loader: CIFAR100Loader,
    device: torch.device,
) -> None:
    mus = collect_mus(model, loader, device)
    score = isotropy_score(mus)
    active = active_units(mus)
    print(f"isotropy_score={score.item():.4f} active_units={active.sum().item()}/{mus.shape[1]}")


def run_probe(
    model: torch.nn.Module,
    loader: CIFAR100Loader,
    device: torch.device,
    num_classes: int = 100,
) -> None:
    mus, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            out = model.forward(images)
            mus.append(out.mu.cpu())
            labels.append(batch_labels)
    z = torch.cat(mus, dim=0)
    y = torch.cat(labels, dim=0)

    probe = LinearProbe(z.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(50):
        optimizer.zero_grad()
        logits = probe(z)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = probe(z)
        top1 = (logits.argmax(dim=-1) == y).float().mean().item()
        top5 = (logits.topk(5, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    print(f"linear_probe top1={top1:.4f} top5={top5:.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = build_eval_loader(args.data_dir, args.batch_size)

    if args.ablation == "isotropy":
        run_isotropy(model, loader, device)
    elif args.ablation == "probe":
        run_probe(model, loader, device)
    elif args.ablation == "param_sweep":
        for p, q, eps in itertools.zip_longest(P_SWEEP, Q_SWEEP, EPS_SWEEP):
            print(
                f"sweep point: p={p} q={q} eps={eps} (re-train with these fixed for full ablation)"
            )
    elif args.ablation == "dim_scaling":
        for d in LATENT_DIM_SWEEP:
            print(f"latent_dim={d} (re-train per dimension for full ablation)")


if __name__ == "__main__":
    main()
