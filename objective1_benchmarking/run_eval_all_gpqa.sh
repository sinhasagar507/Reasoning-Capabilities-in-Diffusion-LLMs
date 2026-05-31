#!/usr/bin/env bash
set -euo pipefail

# ── Resolve paths ───────────────────────────────────────────────────────────────
# Scripts live in objective1_benchmarking/; REPO_ROOT is its parent.
# Artifacts (results/, sft_checkpoints/) live at the repo root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_gpqa_arrow.py"  # updated eval script name
OUTROOT="${REPO_ROOT}/results/gpqa_eval"

# ====== USER SETTINGS ======
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct"
SFT_CKPT="${REPO_ROOT}/sft_checkpoints/llada_mix_temp07/checkpoint-942"  # update if needed

# Run only GPQA (arrow)
DATASETS=("gpqa_diamond")
STEPS=(16 32 64 256)

# Path to your GPQA Arrow dataset (directory containing dataset_info.json OR a .arrow shard)
GPQA_ARROW_DIR="/scratch/ssinha78/main-project/d1/dataset/mix_temp07/diamond_gpqa_test"

# Uploaded image path (keeps the file you showed available for reference)
UPLOADED_IMAGE="/mnt/data/4a5cd763-0d0a-4ab2-b2c1-3d85688817c9.png"

GENLEN=256
BATCH=1
BLOCKLEN=32
TEMP=0.0
CFG=0.0
NPROC=1

# Behavior knobs
MAX_RETRIES="${MAX_RETRIES:-2}"
FORCE="${FORCE:-0}"
START_INDEX="${START_INDEX:-}"

# ====== END USER SETTINGS ======

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[error] eval script not found at ${EVAL_SCRIPT}"
  exit 1
fi

# Validate dataset resources up front
for ds in "${DATASETS[@]}"; do
  case "$ds" in
    gpqa|gpqa_diamond)
      if [[ ! -d "${GPQA_ARROW_DIR}" && ! -f "${GPQA_ARROW_DIR}" ]]; then
        echo "[error] GPQA Arrow path missing: ${GPQA_ARROW_DIR}"
        exit 2
      fi
      ;;
    *)
      echo "[error] unknown dataset in DATASETS: $ds"
      exit 2
      ;;
  esac
done

# ---- helpers ---------------------------------------------------------------
BAR_WIDTH=40
TOTAL=$(( ${#DATASETS[@]} * ${#STEPS[@]} ))

draw_bar () {
  local done=$1 total=$2 width=$3
  local perc=0
  if (( total > 0 )); then perc=$(( 100 * done / total )); fi
  local filled=0
  if (( total > 0 )); then filled=$(( width * done / total )); fi
  local empty=$(( width - filled ))
  printf "\r["
  printf "%0.s#" $(seq 1 $filled)
  printf "%0.s-" $(seq 1 $empty)
  printf "] %3d%%  (%d/%d)" "$perc" "$done" "$total"
}

is_done () {
  local outdir="$1"
  if [[ -f "${outdir}/DONE" ]]; then return 0; fi
  shopt -s nullglob
  local files=("${outdir}"/*_generations.json)
  if (( ${#files[@]} > 0 )) && [[ -s "${files[0]}" ]]; then return 0; fi
  return 1
}

jsonl_lines () {
  local path="$1"
  if [[ -f "$path" ]]; then
    awk 'NF{c++} END{print c+0}' "$path" 2>/dev/null || wc -l < "$path"
  else
    echo 0
  fi
}

pick_master_port () {
  local try_port
  for try_port in $(shuf -i 20000-40000 -n 12); do
    if command -v ss >/dev/null 2>&1; then
      if ! ss -ltn | grep -q ":${try_port} "; then
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT="${try_port}"
        return 0
      fi
    else
      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT="${try_port}"
      return 0
    fi
  done
  : "${MASTER_ADDR:=127.0.0.1}"
  : "${MASTER_PORT:=29500}"
}

run_one () {
  local variant="$1" dataset="$2" steps="$3"
  local outdir="${OUTROOT}/${variant}/${dataset}/g${GENLEN}_s${steps}"
  mkdir -p "$outdir"

  if [[ "$FORCE" != "1" ]] && is_done "$outdir"; then
    echo "[skip] ${variant}/${dataset} steps=${steps} already DONE"
    return 0
  fi

  local ckpt_arg=()
  if [[ "$variant" == "sft" ]]; then
    ckpt_arg=(--checkpoint_path "${SFT_CKPT}")
  fi

  # dataset-specific args (GPQA arrow)
  local ds_args=(--eval_arrow "${GPQA_ARROW_DIR}")

  local jsonl_path="${outdir}/generations.rank0.jsonl"
  local resume_args=(--resume_state "${outdir}/state.json")
  if [[ -n "${START_INDEX}" ]]; then
    resume_args+=(--start_index "${START_INDEX}")
    echo "[resume] Hard override: --start_index=${START_INDEX}"
  else
    local jl; jl=$(jsonl_lines "${jsonl_path}")
    if [[ "${jl}" -gt 0 ]]; then
      resume_args+=(--start_index "${jl}")
      echo "[resume] Detected ${jl} already-streamed lines; nudging --start_index=${jl}"
    fi
  fi

  echo
  echo "[run] ${variant} | ${dataset} | steps=${steps} | gen_len=${GENLEN}"
  echo "      logs → ${outdir}/run.log"

  local attempt=0
  while (( attempt <= MAX_RETRIES )); do
    set +e
    pick_master_port
    stdbuf -oL -eL torchrun --nproc_per_node="${NPROC}" "${EVAL_SCRIPT}" \
      --model_path "${BASE_MODEL}" \
      "${ckpt_arg[@]}" \
      "${ds_args[@]}" \
      --batch_size "${BATCH}" \
      --gen_length "${GENLEN}" \
      --block_length "${BLOCKLEN}" \
      --diffusion_steps "${steps}" \
      --temperature "${TEMP}" \
      --cfg_scale "${CFG}" \
      --output_dir "${outdir}" \
      "${resume_args[@]}" |& tee -a "${outdir}/run.log"
    code=$?
    set -e

    if [[ $code -eq 0 ]] && is_done "$outdir"; then
      date > "${outdir}/DONE"
      return 0
    fi

    echo "[warn] Job failed (exit=${code}) or output missing; attempt $((attempt+1))/${MAX_RETRIES}"
    attempt=$((attempt+1))
    sleep 5
  done

  echo "[fail] ${variant}/${dataset} steps=${steps} after ${MAX_RETRIES} retries"
  return 1
}

# Pre-count DONE to initialize progress bar
DONE_COUNT=0
for ds in "${DATASETS[@]}"; do
  for s in "${STEPS[@]}"; do
    outdir="${OUTROOT}/sft/${ds}/g${GENLEN}_s${s}"
    if is_done "$outdir"; then
      DONE_COUNT=$((DONE_COUNT+1))
    fi
  done
done
draw_bar "$DONE_COUNT" "$TOTAL" "$BAR_WIDTH"

echo
echo "[info] Eval script: ${EVAL_SCRIPT}"
echo "[info] Base (kept for reference): ${BASE_MODEL}"
echo "[info] SFT:         ${SFT_CKPT}"
echo "[info] Datasets:    ${DATASETS[*]}"
echo "[info] Steps:       ${STEPS[*]}"
echo "[info] GENLEN:      ${GENLEN}, BLOCKLEN: ${BLOCKLEN}, BATCH: ${BATCH}"
echo "[info] GPQA Arrow:  ${GPQA_ARROW_DIR}"
echo "[info] Uploaded image (for reference): ${UPLOADED_IMAGE}"

echo "[info] Resuming… already done: ${DONE_COUNT}/${TOTAL}"

# Only run SFT variant
for ds in "${DATASETS[@]}"; do
  for s in "${STEPS[@]}"; do
    run_one "sft" "$ds" "$s" && DONE_COUNT=$((DONE_COUNT+1))
    draw_bar "$DONE_COUNT" "$TOTAL" "$BAR_WIDTH"
  done
done

echo
echo "[done] Outputs under ${OUTROOT}. To force re-run, delete 'DONE' files or run: FORCE=1 bash $(basename "$0")"
