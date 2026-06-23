"""Latent-space geometry evaluations for hypergen VAEs.

Evaluations:
  cosine_sim   — intra- vs inter-class cosine similarity
  knn          — k-nearest-neighbor accuracy (k=5,10,20)
  fid          — FID between real and reconstructed test images
  pca          — PCA projection of latent codes, saved as PNG

Usage:
    uv run python apps/eval_geometry.py --model tnbbeta \
        --checkpoint checkpoints/tnbbeta/tnbbeta_epoch200.pt --eval cosine_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from hypergen.baselines import VMFVAE, GaussianVAE, PowerSphericalVAE
from hypergen.models import TNBbetaVAE

torch.backends.cudnn.enabled = False

matplotlib.use("Agg")

MODEL_REGISTRY = {
    "gaussian": GaussianVAE,
    "vmf": VMFVAE,
    "power_spherical": PowerSphericalVAE,
    "tnbbeta": TNBbetaVAE,
}

CIFAR100Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent geometry evaluations.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="tnbbeta")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--eval",
        choices=["cosine_sim", "knn", "fid", "pca", "all"],
        default="all",
    )
    return parser.parse_args()


def build_eval_loader(data_dir: Path, batch_size: int) -> CIFAR100Loader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def collect_mus_and_labels(
    model: torch.nn.Module, loader: CIFAR100Loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    all_mus, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        out = model.forward(images)
        all_mus.append(out.mu.cpu())
        all_labels.append(labels)
    return torch.cat(all_mus, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def collect_recons(
    model: torch.nn.Module, loader: CIFAR100Loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    all_real, all_recon = [], []
    for images, _labels in loader:
        images = images.to(device)
        out = model.forward(images)
        all_real.append(images.cpu())
        all_recon.append(torch.sigmoid(out.recon).cpu())
    return torch.cat(all_real, dim=0), torch.cat(all_recon, dim=0)


def run_cosine_sim(mus: torch.Tensor, labels: torch.Tensor) -> None:
    mus_norm = mus / mus.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    classes = labels.unique()
    intra_sims, inter_sims = [], []

    for c in classes:
        mask = labels == c
        class_mus = mus_norm[mask]
        if class_mus.shape[0] < 2:
            continue
        sim_matrix = class_mus @ class_mus.T
        n = class_mus.shape[0]
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        intra_sims.append(sim_matrix[triu_mask].mean().item())

    for _ in range(200):
        c1, c2 = classes[torch.randint(len(classes), (2,))]
        if c1 == c2:
            continue
        m1 = mus_norm[labels == c1]
        m2 = mus_norm[labels == c2]
        idx1 = torch.randint(m1.shape[0], (min(50, m1.shape[0]),))
        idx2 = torch.randint(m2.shape[0], (min(50, m2.shape[0]),))
        inter_sims.append((m1[idx1] @ m2[idx2].T).mean().item())

    intra_mean = np.mean(intra_sims)
    intra_std = np.std(intra_sims)
    inter_mean = np.mean(inter_sims)
    inter_std = np.std(inter_sims)
    separation = intra_mean - inter_mean

    print(f"intra_class_cosine_sim={intra_mean:.4f} ± {intra_std:.4f}")
    print(f"inter_class_cosine_sim={inter_mean:.4f} ± {inter_std:.4f}")
    print(f"separation_gap={separation:.4f}")


def run_knn(mus: torch.Tensor, labels: torch.Tensor) -> None:
    mus_norm = mus / mus.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n = mus_norm.shape[0]
    chunk_size = 500
    for k in [5, 10, 20]:
        correct = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = mus_norm[start:end]
            sims = chunk @ mus_norm.T
            sims[torch.arange(end - start), torch.arange(start, end)] = -2.0
            _, topk_idx = sims.topk(k, dim=1)
            topk_labels = labels[topk_idx]
            preds = topk_labels.mode(dim=1).values
            correct += (preds == labels[start:end]).sum().item()
        acc = correct / n
        print(f"knn_k{k}_accuracy={acc:.4f}")


def run_fid(model: torch.nn.Module, loader: CIFAR100Loader, device: torch.device) -> None:
    real, recon = collect_recons(model, loader, device)

    resize = transforms.Resize((299, 299), antialias=True)
    inception = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.DEFAULT
    )
    inception.fc = torch.nn.Identity()  # type: ignore[assignment]
    inception.eval()
    inception.to(device)

    def get_activations(images: torch.Tensor) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:  # type: ignore[type-var]
        acts = []
        for i in range(0, images.shape[0], 64):
            batch = resize(images[i : i + 64]).to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            with torch.no_grad():
                a = inception(batch)
            acts.append(a.cpu().numpy())
        return np.concatenate(acts, axis=0)

    act_real = get_activations(real)
    act_recon = get_activations(recon)

    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_recon, sigma_recon = act_recon.mean(axis=0), np.cov(act_recon, rowvar=False)

    diff = mu_real - mu_recon
    from scipy import linalg

    covmean, _ = linalg.sqrtm(sigma_real @ sigma_recon, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_real + sigma_recon - 2.0 * covmean))
    print(f"fid={fid:.2f}")


def run_pca(mus: torch.Tensor, labels: torch.Tensor, model_name: str, output_dir: Path) -> None:
    mus_centered = mus - mus.mean(dim=0)
    _, _, vh = torch.linalg.svd(mus_centered, full_matrices=False)
    proj = (mus_centered @ vh[:2].T).numpy()
    labels_np = labels.numpy()

    superclass_map = [
        4,
        1,
        14,
        8,
        0,
        6,
        7,
        7,
        18,
        3,
        3,
        14,
        9,
        18,
        7,
        11,
        3,
        9,
        7,
        11,
        6,
        11,
        5,
        10,
        7,
        6,
        13,
        15,
        3,
        15,
        0,
        11,
        1,
        10,
        12,
        14,
        16,
        17,
        5,
        0,
        2,
        4,
        17,
        8,
        14,
        5,
        2,
        17,
        4,
        1,
        1,
        10,
        16,
        15,
        9,
        0,
        12,
        12,
        8,
        5,
        2,
        4,
        6,
        10,
        15,
        2,
        17,
        8,
        16,
        12,
        1,
        9,
        16,
        12,
        8,
        13,
        13,
        13,
        16,
        3,
        9,
        5,
        6,
        15,
        13,
        0,
        11,
        4,
        0,
        18,
        14,
        6,
        11,
        9,
        1,
        10,
        13,
        2,
        17,
        3,
    ]
    superclass_names = [
        "aquatic mammals",
        "fish",
        "flowers",
        "food containers",
        "fruit & veg",
        "household electrical",
        "household furniture",
        "insects",
        "large carnivores",
        "large outdoor man-made",
        "large outdoor natural",
        "large omnivores & herbivores",
        "medium mammals",
        "non-insect invertebrates",
        "people",
        "reptiles",
        "small mammals",
        "trees",
        "vehicles 1",
        "vehicles 2",
    ]
    super_labels = np.array([superclass_map[c] for c in labels_np])
    n_super = 20

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", n_super)
    for sc in range(n_super):
        mask = super_labels == sc
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=[cmap(sc)],
            s=3,
            alpha=0.5,
            label=superclass_names[sc],
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of latent codes — {model_name}")
    ax.legend(
        fontsize=6,
        markerscale=3,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pca_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = build_eval_loader(args.data_dir, args.batch_size)
    run_all = args.eval == "all"

    if run_all or args.eval == "cosine_sim" or args.eval == "knn" or args.eval == "pca":
        mus, labels = collect_mus_and_labels(model, loader, device)

    if run_all or args.eval == "cosine_sim":
        print(f"=== cosine similarity ({args.model}) ===")
        run_cosine_sim(mus, labels)

    if run_all or args.eval == "knn":
        print(f"=== kNN accuracy ({args.model}) ===")
        run_knn(mus, labels)

    if run_all or args.eval == "fid":
        print(f"=== FID ({args.model}) ===")
        run_fid(model, loader, device)

    if run_all or args.eval == "pca":
        print(f"=== PCA ({args.model}) ===")
        run_pca(mus, labels, args.model, args.output_dir)


if __name__ == "__main__":
    main()
