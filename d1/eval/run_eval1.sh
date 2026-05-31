#!/usr/bin/env bash
# Robust full evaluation runner for updated eval.py

set -euo pipefail

# -----------------------------
# 1) CONFIG (edit if needed)
# -----------------------------
# GPUs to use (comma-separated). You can override via: bash run_eval.sh 0,1
GPU_LIST="${1:-0,1}"

# Base model repo id on HF Hub (already used in your training/eval code)
BASE_MODEL="${2:-GSAI-ML/LLaDA-8B-Instruct}"

# Root directory containing your SFT outputs (adapter checkpoints live under checkpoint-*)
ADAPTER_ROOT="${3:-/scratch/ayada127/checkpoints/sft-llada-8b}"

# Batch size for evaluation
BATCH_SIZE="${BATCH_SIZE:-8}"

# Datasets and generation lengths to run
DATASETS=(${DATASETS:-countdown sudoku math gsm8k})
LENGTHS=(${LENGTHS:-128 256})

# Hugging Face cache on scratch (avoids home quota and speedups)
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# NCCL / CUDA env (safe defaults on many clusters)
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# If you ever see NCCL fabric/P2P issues, uncomment one or both:
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1

# -----------------------------
# 2) DERIVED PATHS
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/eval.py"

# Timestamped results directory under eval/
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/results/${STAMP}"
mkdir -p "${OUT_DIR}"

# Latest adapter checkpoint (if present)
LATEST_ADAPTER="$(ls -1dt "${ADAPTER_ROOT}"/checkpoint-* 2>/dev/null | head -1 || true)"
CHECKPOINT_ARG=()
if [[ -n "${LATEST_ADAPTER}" && -d "${LATEST_ADAPTER}" ]]; then
  CHECKPOINT_ARG=(--checkpoint_path "${LATEST_ADAPTER}")
else
  echo "[WARN] No adapter checkpoints found under: ${ADAPTER_ROOT}/checkpoint-*"
  echo "       Proceeding with **base model only** (no --checkpoint_path)."
fi

# Number of processes from GPU list
NPROC="$(awk -F',' '{print NF}' <<<"${GPU_LIST}")"

# Quick sanity prints
echo "================ EVAL CONFIG ================"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node      : ${NPROC}"
echo "Base model (HF)     : ${BASE_MODEL}"
echo "Adapter checkpoint  : ${LATEST_ADAPTER:-<none – base model only>}"
echo "Results dir         : ${OUT_DIR}"
echo "Datasets            : ${DATASETS[*]}"
echo "Gen lengths         : ${LENGTHS[*]}"
echo "Batch size          : ${BATCH_SIZE}"
echo "HF_HOME             : ${HF_HOME}"
echo "TRANSFORMERS_CACHE  : ${TRANSFORMERS_CACHE}"
echo "============================================="

# Optional: show CUDA device count as seen by torch
python - <<'PY'
import torch, os
print(f"[INFO] torch.cuda.device_count() = {torch.cuda.device_count()} "
      f"(CUDA_VISIBLE_DEVICES='{os.getenv('CUDA_VISIBLE_DEVICES','')}')")
PY

# -----------------------------
# 3) RUN ALL EVALS
# -----------------------------
for D in "${DATASETS[@]}"; do
  for L in "${LENGTHS[@]}"; do
    echo ""
    echo ">>> Running dataset=${D}, gen_length=${L}"
    torchrun --nproc_per_node="${NPROC}" "${PY}" \
      --model_path "${BASE_MODEL}" \
      "${CHECKPOINT_ARG[@]}" \
      --dataset "${D}" \
      --gen_length "${L}" \
      --batch_size "${BATCH_SIZE}" \
      --output_dir "${OUT_DIR}"
  done
done

echo ""
echo "All evaluations completed."
echo "Saved JSONs under: ${OUT_DIR}"
ls -lh "${OUT_DIR}" || true
