#!/usr/bin/env bash
# Train all four VAE variants in parallel on 4 A100 GPUs via SLURM.
#
# Usage:
#   sbatch scripts/train_all.sh
#
# Override defaults at submission time, e.g.:
#   sbatch --time=8:00:00 scripts/train_all.sh

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_train
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
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
mkdir -p logs checkpoints

uv sync

# ── parse checkpoint root from args (default: checkpoints/) ───────────────────
CKPT_ROOT="checkpoints"
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            CKPT_ROOT="$2"; shift 2 ;;
        *)
            PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# ── train all four models in parallel, one per GPU ────────────────────────────
MODELS=(gaussian vmf power_spherical tnbbeta)
PIDS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    echo "[$(date)] Starting ${model} on GPU ${i}"
    CUDA_VISIBLE_DEVICES="${i}" uv run --no-sync python apps/train_cifar100.py \
        --model "${model}" \
        --latent-dim 64 \
        --beta 1.0 \
        --epochs 200 \
        --batch-size 256 \
        --seed 0 \
        --checkpoint-dir "${CKPT_ROOT}/${model}" \
        "${PASSTHROUGH_ARGS[@]}" \
        > "logs/${model}_${SLURM_JOB_ID}.out" \
        2> "logs/${model}_${SLURM_JOB_ID}.err" &
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
