#!/usr/bin/env bash
# Sweep q_max values for TNBbeta VAE: train in parallel, eval each when done.
#
# Usage:
#   sbatch scripts/sweep_qmax.sh
#   sbatch scripts/sweep_qmax.sh --checkpoint-dir /scratch/ckpt --data-dir /scratch/data

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=hypergen_qmax
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:7
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_DISABLED=1

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints plots

uv sync

# ── parse args ────────────────────────────────────────────────────────────────
CKPT_ROOT="checkpoints"
DATA_DIR="data"
EPOCHS=200
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        *)                shift ;;
    esac
done

# ── sweep config ──────────────────────────────────────────────────────────────
Q_MAX_VALUES=(0.10 0.15 0.20 0.25 0.30 0.40 0.50)

# ── function: run all evals for a single q_max ────────────────────────────────
run_evals() {
    local qmax="$1"
    local ckpt_dir="${CKPT_ROOT}/tnbbeta_qmax${qmax}"
    local ckpt="${ckpt_dir}/tnbbeta_epoch${EPOCHS}.pt"

    echo "[$(date)] === EVALS for q_max=${qmax} ==="

    # geometry evals (cosine_sim, knn, fid, ood)
    for eval_type in cosine_sim knn fid ood; do
        echo "[$(date)]   ${eval_type} (q_max=${qmax})"
        uv run --no-sync python apps/eval_geometry.py \
            --model tnbbeta \
            --checkpoint "${ckpt}" \
            --latent-dim 64 \
            --eval "${eval_type}" \
            --data-dir "${DATA_DIR}" \
            --fixed-eps 1.0 \
            --q-max "${qmax}"
    done

    # probe
    echo "[$(date)]   probe (q_max=${qmax})"
    uv run --no-sync python apps/eval_latent.py \
        --model tnbbeta \
        --checkpoint "${ckpt}" \
        --latent-dim 64 \
        --ablation probe \
        --data-dir "${DATA_DIR}"

    # param stats
    echo "[$(date)]   param_stats (q_max=${qmax})"
    uv run --no-sync python apps/eval_geometry.py \
        --model tnbbeta \
        --checkpoint "${ckpt}" \
        --latent-dim 64 \
        --eval param_stats \
        --data-dir "${DATA_DIR}" \
        --fixed-eps 1.0 \
        --q-max "${qmax}"

    echo "[$(date)] === DONE evals for q_max=${qmax} ==="
}
export -f run_evals
export CKPT_ROOT DATA_DIR EPOCHS

# ── launch training for each q_max in parallel ───────────────────────────────
PIDS=()

for i in "${!Q_MAX_VALUES[@]}"; do
    qmax="${Q_MAX_VALUES[$i]}"
    ckpt_dir="${CKPT_ROOT}/tnbbeta_qmax${qmax}"
    echo "[$(date)] Starting q_max=${qmax} on GPU ${i}"
    (
        CUDA_VISIBLE_DEVICES="${i}" uv run --no-sync python apps/train_cifar100.py \
            --model tnbbeta \
            --latent-dim 64 \
            --beta 1.0 \
            --epochs "${EPOCHS}" \
            --batch-size 256 \
            --seed 0 \
            --data-dir "${DATA_DIR}" \
            --checkpoint-dir "${ckpt_dir}" \
            --fixed-eps 1.0 \
            --q-max "${qmax}" \
            > "logs/tnbbeta_qmax${qmax}_${SLURM_JOB_ID}.out" \
            2> "logs/tnbbeta_qmax${qmax}_${SLURM_JOB_ID}.err" \
        && CUDA_VISIBLE_DEVICES="${i}" run_evals "${qmax}" \
            >> "logs/tnbbeta_qmax${qmax}_${SLURM_JOB_ID}.out" \
            2>> "logs/tnbbeta_qmax${qmax}_${SLURM_JOB_ID}.err"
    ) &
    PIDS+=($!)
done

# ── wait for all and report ───────────────────────────────────────────────────
FAILED=0
for i in "${!Q_MAX_VALUES[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[$(date)] FAILED: q_max=${Q_MAX_VALUES[$i]}"
        FAILED=1
    else
        echo "[$(date)] DONE: q_max=${Q_MAX_VALUES[$i]}"
    fi
done

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Check individual logs for metrics:"
for qmax in "${Q_MAX_VALUES[@]}"; do
    echo "  q_max=${qmax}: logs/tnbbeta_qmax${qmax}_${SLURM_JOB_ID}.out"
done

exit "${FAILED}"
