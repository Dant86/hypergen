"""Latent-space ablations for hypergen VAEs.

Usage:
    uv run python apps/eval_latent.py --checkpoint checkpoints/tnbbeta_epoch200.pt --model tnbbeta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from hypergen.baselines import VMFVAE, GaussianVAE, PowerSphericalVAE
from hypergen.metrics import LinearProbe, active_units, isotropy_score
from hypergen.models import TNBbetaVAE

torch.backends.cudnn.enabled = False

MODEL_REGISTRY = {
    "gaussian": GaussianVAE,
    "vmf": VMFVAE,
    "power_spherical": PowerSphericalVAE,
    "tnbbeta": TNBbetaVAE,
}

BETA_SWEEP = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
LATENT_DIM_SWEEP = [16, 32, 64, 128, 256]
SWEEP_EPOCHS = 50

CIFAR100Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-space ablations.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="tnbbeta")
    parser.add_argument("--checkpoint", type=Path, default=None)
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


def _train_short(
    model: torch.nn.Module,
    loader: CIFAR100Loader,
    device: torch.device,
    epochs: int,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for images, _labels in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss, out = model.loss(images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            kl_val = out.kl.item() if out is not None else 0.0
            print(f"  epoch {epoch:3d}  loss={epoch_loss / n_batches:.2f}  kl={kl_val:.2f}")


def _eval_model(
    model: torch.nn.Module, loader: CIFAR100Loader, device: torch.device
) -> tuple[float, int, int]:
    model.eval()
    mus = collect_mus(model, loader, device)
    score = isotropy_score(mus)
    active = active_units(mus)
    return score.item(), int(active.sum().item()), mus.shape[1]


def run_param_sweep(data_dir: Path, batch_size: int, latent_dim: int, device: torch.device) -> None:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    train_loader: CIFAR100Loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )
    eval_loader: CIFAR100Loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"{'beta':>6} {'isotropy':>10} {'active_units':>14}")
    for beta in BETA_SWEEP:
        torch.manual_seed(0)
        model = TNBbetaVAE(latent_dim=latent_dim, beta=beta).to(device)
        _train_short(model, train_loader, device, SWEEP_EPOCHS)
        score, active, total = _eval_model(model, eval_loader, device)
        print(f"{beta:6.2f} {score:10.4f} {active:>6}/{total}")


def run_dim_scaling(data_dir: Path, batch_size: int, device: torch.device) -> None:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    train_loader: CIFAR100Loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    eval_transform = transforms.Compose([transforms.ToTensor()])
    eval_dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )
    eval_loader: CIFAR100Loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"{'dim':>6} {'isotropy':>10} {'active_units':>14}")
    for d in LATENT_DIM_SWEEP:
        torch.manual_seed(0)
        model = TNBbetaVAE(latent_dim=d, beta=1.0).to(device)
        _train_short(model, train_loader, device, SWEEP_EPOCHS)
        score, active, total = _eval_model(model, eval_loader, device)
        print(f"{d:6d} {score:10.4f} {active:>6}/{total}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation == "param_sweep":
        run_param_sweep(args.data_dir, args.batch_size, args.latent_dim, device)
        return
    if args.ablation == "dim_scaling":
        run_dim_scaling(args.data_dir, args.batch_size, device)
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for isotropy/probe ablations")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = build_eval_loader(args.data_dir, args.batch_size)

    if args.ablation == "isotropy":
        run_isotropy(model, loader, device)
    elif args.ablation == "probe":
        run_probe(model, loader, device)


if __name__ == "__main__":
    main()
