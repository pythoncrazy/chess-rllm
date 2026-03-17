#!/bin/bash
# Generate train and val datasets.
#
# Usage:
#   bash generate_data.sh [train_out] [val_out] [n_train] [n_val]
#
# Defaults:
#   train_out = data/sf_train.jsonl
#   val_out   = data/sf_val.jsonl
#   n_train   = 500000
#   n_val     = 5000

set -euo pipefail

DIR=/home/supersketchy/git/chess-rllm
TRAIN_OUT=${1:-${DIR}/data/sf_train.jsonl}
VAL_OUT=${2:-${DIR}/data/sf_val.jsonl}
N_TRAIN=${3:-10000000}
N_VAL=${4:-10000}

cd "${DIR}"

echo "[$(date)] Generating training data (n=${N_TRAIN})..."
uv run python datagen.py --n "${N_TRAIN}" --out "${TRAIN_OUT}"

echo "[$(date)] Generating val data (n=${N_VAL})..."
uv run python datagen.py --n "${N_VAL}" --out "${VAL_OUT}"

echo "[$(date)] Done."
echo "  Train: ${TRAIN_OUT}"
echo "  Val:   ${VAL_OUT}"
