#!/usr/bin/env bash
# Generate SLERP interpolation grids for all four models sequentially.
#
# Usage:
#   sbatch scripts/plot_all.sh
#   sbatch scripts/plot_all.sh --checkpoint-dir /scratch/checkpoints --data-dir /scratch/data

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_plot
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
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
LATENT_DIM=64
EPOCHS=200
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CKPT_ROOT="$2"; shift 2 ;;
        --latent-dim)     LATENT_DIM="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        *)                PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# ── plot all four models sequentially ─────────────────────────────────────────
MODELS=(gaussian vmf power_spherical tnbbeta)

for model in "${MODELS[@]}"; do
    ckpt="${CKPT_ROOT}/${model}/${model}_epoch${EPOCHS}.pt"
    echo "[$(date)] Plotting SLERP grid for ${model}"
    uv run --no-sync python apps/plot_results.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --latent-dim "${LATENT_DIM}" \
        --output "plots/slerp_${model}.png" \
        "${PASSTHROUGH_ARGS[@]}"
    echo "[$(date)] DONE: ${model}"
done
