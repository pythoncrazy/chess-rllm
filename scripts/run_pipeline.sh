#!/bin/bash
# Run SFT then GRPO from the resulting checkpoint.
#
# Usage:
#   bash run_pipeline.sh <sft_run_name> <grpo_run_name> [train_jsonl] [val_jsonl]
#
# Defaults:
#   train_jsonl = data/sf_train.jsonl
#   val_jsonl   = data/sf_val.jsonl
#
# Example:
#   bash run_pipeline.sh uci-v2 uci-grpo-v2

set -euo pipefail

SFT_RUN=${1:?Usage: $0 <sft_run_name> <grpo_run_name>}
GRPO_RUN=${2:?}

DIR=/home/supersketchy/git/chess-rllm
TRAIN=${3:-${DIR}/data/sf_train.jsonl}
VAL=${4:-${DIR}/data/sf_val.jsonl}

SFT_LOG=${DIR}/outputs/sft-${SFT_RUN}.log
GRPO_LOG=${DIR}/outputs/grpo-${GRPO_RUN}.log
CKPT_DIR=${DIR}/checkpoints/sft/${SFT_RUN}/checkpoints.jsonl

mkdir -p "${DIR}/outputs"
cd "${DIR}"

# --- SFT ---
echo "[$(date)] Starting SFT run '${SFT_RUN}'..." | tee -a "${SFT_LOG}"
uv run python sft_train.py \
    "run_name=${SFT_RUN}" \
    "data.train_files=${TRAIN}" \
    "data.val_files=${VAL}" \
    2>&1 | tee -a "${SFT_LOG}"

# --- Extract checkpoint ---
echo "[$(date)] SFT complete. Extracting checkpoint..." | tee -a "${GRPO_LOG}"
FINAL_CKPT=$(grep '"final"' "${CKPT_DIR}" 2>/dev/null \
    | uv run python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip())['state_path'])" 2>/dev/null)

if [ -z "${FINAL_CKPT}" ]; then
    FINAL_CKPT=$(tail -1 "${CKPT_DIR}" \
        | uv run python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip())['state_path'])")
fi

echo "[$(date)] Using checkpoint: ${FINAL_CKPT}" | tee -a "${GRPO_LOG}"

# --- GRPO ---
echo "[$(date)] Starting GRPO run '${GRPO_RUN}'..." | tee -a "${GRPO_LOG}"
uv run python train.py \
    "run_name=${GRPO_RUN}" \
    "training.resume_from_tinker_id=${FINAL_CKPT}" \
    "train_files=${TRAIN}" \
    "val_files=${VAL}" \
    2>&1 | tee -a "${GRPO_LOG}"

echo "[$(date)] Pipeline complete." | tee -a "${GRPO_LOG}"
