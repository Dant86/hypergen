#!/usr/bin/env bash
# Generate SLERP interpolation grids for all four models in parallel.
#
# Usage:
#   sbatch scripts/plot_all.sh
#   sbatch scripts/plot_all.sh --checkpoint-dir /scratch/checkpoints --data-dir /scratch/data

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_plot
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# ── plot all four models in parallel, one per GPU ─────────────────────────────
MODELS=(gaussian vmf power_spherical tnbbeta)
PIDS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    ckpt="${CKPT_ROOT}/${model}/${model}_epoch${EPOCHS}.pt"
    echo "[$(date)] Plotting SLERP grid for ${model} on GPU ${i}"
    CUDA_VISIBLE_DEVICES="${i}" uv run --no-sync python apps/plot_results.py \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --latent-dim "${LATENT_DIM}" \
        --output "plots/slerp_${model}.png" \
        "${PASSTHROUGH_ARGS[@]}" \
        > "logs/plot_${model}_${SLURM_JOB_ID}.out" \
        2> "logs/plot_${model}_${SLURM_JOB_ID}.err" &
    PIDS+=($!)
done

# ── wait for all and report ───────────────────────────────────────────────────
FAILED=0
for i in "${!MODELS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[$(date)] FAILED: ${MODELS[$i]} (PID ${PIDS[$i]})"
        FAILED=1
    else
        echo "[$(date)] DONE: ${MODELS[$i]}"
    fi
done

exit "${FAILED}"
