#!/bin/bash
# Generate train and val datasets.
#
# Usage:
#   bash generate_data.sh [train_out] [val_out] [n_train] [n_val]
#
# Defaults:
#   train_out = data/sf_train_10m.jsonl
#   val_out   = data/sf_val_10k.jsonl
#   n_train   = 500000
#   n_val     = 5000

set -euo pipefail

DIR=/home/supersketchy/git/chess-rllm
TRAIN_OUT=${1:-${DIR}/data/sf_train_10m.jsonl}
VAL_OUT=${2:-${DIR}/data/sf_val_10k.jsonl}
N_TRAIN=${3:-10000000}
N_VAL=${4:-10000}

mkdir -p "${DIR}/outputs/datagen"
cd "${DIR}"

echo "[$(date)] Generating training data (n=${N_TRAIN})..."
uv run python src/datagen.py --n "${N_TRAIN}" --out "${TRAIN_OUT}" \
    2>&1 | tee "${DIR}/outputs/datagen/train-$(date +%Y%m%d-%H%M%S).log"

echo "[$(date)] Generating val data (n=${N_VAL})..."
uv run python src/datagen.py --n "${N_VAL}" --out "${VAL_OUT}" \
    2>&1 | tee "${DIR}/outputs/datagen/val-$(date +%Y%m%d-%H%M%S).log"

echo "[$(date)] Done."
echo "  Train: ${TRAIN_OUT}"
echo "  Val:   ${VAL_OUT}"
