"""Trains a TNBbeta-VAE (or baseline) on CIFAR-100.

Usage:
    uv run python apps/train_cifar100.py --model tnbbeta --latent-dim 64 --beta 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from hypergen.baselines import VMFVAE, GaussianVAE, PowerSphericalVAE
from hypergen.models import TNBbetaVAE

torch.backends.cudnn.enabled = False

LATENT_DIM = 64
BATCH_SIZE = 256
EPOCHS = 200
LR = 3e-4
WEIGHT_DECAY = 1e-4
BETA = 1.0

MODEL_REGISTRY = {
    "gaussian": GaussianVAE,
    "vmf": VMFVAE,
    "power_spherical": PowerSphericalVAE,
    "tnbbeta": TNBbetaVAE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hyperspherical VAE on CIFAR-100.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="tnbbeta")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb.")
    parser.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    return parser.parse_args()


def build_dataloader(
    data_dir: Path, batch_size: int, dataset: str = "cifar100"
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    ds_cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = ds_cls(root=str(data_dir), train=True, download=True, transform=train_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(latent_dim=args.latent_dim, beta=args.beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loader = build_dataloader(args.data_dir, args.batch_size, args.dataset)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="hypergen", config=vars(args))

    for epoch in range(args.epochs):
        out = None
        for images, _labels in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss, out = model.loss(images)
            loss.backward()
            optimizer.step()

        if out is None:
            continue
        active = (out.mu.var(dim=0) > 0.01).float().mean().item() if hasattr(out, "mu") else 0.0
        print(
            f"epoch={epoch} elbo={out.elbo.item():.4f} "
            f"recon={out.recon_loss.item():.4f} kl={out.kl.item():.4f} active_frac={active:.3f}"
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "elbo": out.elbo.item(),
                    "recon_loss": out.recon_loss.item(),
                    "kl": out.kl.item(),
                    "active_units_frac": active,
                }
            )

        if (epoch + 1) % 10 == 0:
            ckpt_path = args.checkpoint_dir / f"{args.model}_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
