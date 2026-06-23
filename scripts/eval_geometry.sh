#!/usr/bin/env bash
# Run geometry evaluations for all four models sequentially.
#
# Usage:
#   sbatch scripts/eval_geometry.sh
#   sbatch scripts/eval_geometry.sh --eval cosine_sim
#   sbatch scripts/eval_geometry.sh --checkpoint-dir /scratch/checkpoints

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_geom
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
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
EVAL_TYPE="all"
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CKPT_ROOT="$2"; shift 2 ;;
        --latent-dim)     LATENT_DIM="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --eval)           EVAL_TYPE="$2"; shift 2 ;;
        *)                PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# ── eval all four models sequentially ─────────────────────────────────────────
MODELS=(gaussian vmf power_spherical tnbbeta)

for model in "${MODELS[@]}"; do
    ckpt="${CKPT_ROOT}/${model}/${model}_epoch${EPOCHS}.pt"
    echo "[$(date)] Geometry eval: ${model} (${EVAL_TYPE})"
    uv run --no-sync python apps/eval_geometry.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --latent-dim "${LATENT_DIM}" \
        --eval "${EVAL_TYPE}" \
        "${PASSTHROUGH_ARGS[@]}"
    echo "[$(date)] DONE: ${model}"
done
