#!/bin/bash
# Full pipeline: datagen → val datagen → SFT → GRPO
#
# Usage:
#   bash scripts/run_full_pipeline.sh <sft_run_name> <grpo_run_name>
#
# Example:
#   bash scripts/run_full_pipeline.sh uci-v6 uci-grpo-v6

set -euo pipefail

SFT_RUN=${1:?Usage: $0 <sft_run_name> <grpo_run_name>}
GRPO_RUN=${2:?}

DIR=/home/supersketchy/git/chess-rllm
TRAIN=${DIR}/data/sf_train_10m.jsonl
VAL=${DIR}/data/sf_val_10k.jsonl

SFT_LOG=${DIR}/outputs/train/sft-${SFT_RUN}.log
GRPO_LOG=${DIR}/outputs/train/grpo-${GRPO_RUN}.log

mkdir -p "${DIR}/outputs/train" "${DIR}/outputs/datagen" "${DIR}/outputs/eval"
cd "${DIR}"

# --- Wait for datagen (train) ---
echo "[$(date)] Waiting for train datagen to finish (data/sf_train_10m.jsonl)..."
while pgrep -f "src/datagen.py.*sf_train_10m" > /dev/null 2>&1; do
    ROWS=$(wc -l < "${TRAIN}" 2>/dev/null || echo 0)
    echo "[$(date)]   train rows so far: ${ROWS}"
    sleep 60
done
TRAIN_ROWS=$(wc -l < "${TRAIN}")
echo "[$(date)] Train datagen done: ${TRAIN_ROWS} rows in ${TRAIN}"

# --- Generate val data ---
if [ ! -f "${VAL}" ]; then
    echo "[$(date)] Generating val data (n=10000)..."
    uv run python src/datagen.py --n 10000 --out "${VAL}" \
        --game-time 0.1 --analysis-time 0.1 \
        2>&1 | tee -a "${SFT_LOG}"
    echo "[$(date)] Val datagen done: $(wc -l < "${VAL}") rows"
else
    echo "[$(date)] Val data already exists: $(wc -l < "${VAL}") rows"
fi

# --- SFT ---
echo "[$(date)] Starting SFT run '${SFT_RUN}'..." | tee -a "${SFT_LOG}"
uv run python src/sft_train.py \
    "run_name=${SFT_RUN}" \
    "data.train_files=${TRAIN}" \
    "data.val_files=${VAL}" \
    2>&1 | tee -a "${SFT_LOG}"
echo "[$(date)] SFT complete." | tee -a "${SFT_LOG}"

# --- Extract SFT checkpoint ---
CKPT_DIR=${DIR}/checkpoints/sft/${SFT_RUN}/checkpoints.jsonl
echo "[$(date)] Extracting SFT checkpoint from ${CKPT_DIR}..." | tee -a "${GRPO_LOG}"
FINAL_CKPT=$(grep '"final"' "${CKPT_DIR}" 2>/dev/null \
    | uv run python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip())['state_path'])" 2>/dev/null)

if [ -z "${FINAL_CKPT}" ]; then
    FINAL_CKPT=$(tail -1 "${CKPT_DIR}" \
        | uv run python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip())['state_path'])")
fi
echo "[$(date)] Using checkpoint: ${FINAL_CKPT}" | tee -a "${GRPO_LOG}"

# --- GRPO ---
echo "[$(date)] Starting GRPO run '${GRPO_RUN}'..." | tee -a "${GRPO_LOG}"
uv run python src/train.py \
    "run_name=${GRPO_RUN}" \
    "training.resume_from_tinker_id=${FINAL_CKPT}" \
    "train_files=${TRAIN}" \
    "val_files=${VAL}" \
    2>&1 | tee -a "${GRPO_LOG}"

echo "[$(date)] Pipeline complete." | tee -a "${GRPO_LOG}"
