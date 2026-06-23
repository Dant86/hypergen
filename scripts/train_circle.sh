#!/usr/bin/env bash
# Train all four VAE variants with latent_dim=2, then plot on S^1.
#
# Usage:
#   sbatch scripts/train_circle.sh
#   sbatch scripts/train_circle.sh --data-dir /scratch/data --checkpoint-dir /scratch/ckpt

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_circle
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TORCH_CUDNN_V8_API_DISABLED=1

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs plots

uv sync

# ── parse args ────────────────────────────────────────────────────────────────
CKPT_ROOT="checkpoints"
DATA_DIR="data"
EPOCHS=200
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        *)                PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

LATENT_DIM=2
MODELS=(gaussian vmf power_spherical tnbbeta)

# ── train all four models in parallel, one per GPU ───────────────────────────
PIDS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    ckpt_dir="${CKPT_ROOT}/${model}_d2"
    echo "[$(date)] Starting ${model} on GPU ${i} (d=${LATENT_DIM})"
    CUDA_VISIBLE_DEVICES="${i}" uv run --no-sync python apps/train_cifar100.py \
        --model "${model}" \
        --latent-dim "${LATENT_DIM}" \
        --beta 1.0 \
        --epochs "${EPOCHS}" \
        --batch-size 256 \
        --seed 0 \
        --data-dir "${DATA_DIR}" \
        --checkpoint-dir "${ckpt_dir}" \
        --dataset cifar10 \
        "${PASSTHROUGH_ARGS[@]}" \
        > "logs/${model}_d2_${SLURM_JOB_ID}.out" \
        2> "logs/${model}_d2_${SLURM_JOB_ID}.err" &
    PIDS+=($!)
done

FAILED=0
for i in "${!MODELS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[$(date)] FAILED: ${MODELS[$i]} (PID ${PIDS[$i]})"
        FAILED=1
    else
        echo "[$(date)] DONE training: ${MODELS[$i]}"
    fi
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo "Training failed, skipping plots"
    exit 1
fi

# ── generate circle plots on CIFAR-10 ───────────────────────────────────────
for model in "${MODELS[@]}"; do
    ckpt="${CKPT_ROOT}/${model}_d2/${model}_epoch${EPOCHS}.pt"
    echo "[$(date)] Circle plot: ${model}"
    uv run --no-sync python apps/eval_geometry.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --latent-dim "${LATENT_DIM}" \
        --data-dir "${DATA_DIR}" \
        --dataset cifar10 \
        --eval circle
    echo "[$(date)] DONE plot: ${model}"
done
