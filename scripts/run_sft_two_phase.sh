#!/bin/bash
# Two-phase SFT:
#   Phase 1 — 500k chess positions (sf_train_10m.jsonl), normal batch/LR
#   Phase 2 — English explanations (english_50k.jsonl), small batch/LR
#             resumes automatically from Phase 1's final checkpoint
set -euo pipefail

PHASE1_DIR="checkpoints/sft/phase1-chess-500k"
mkdir -p outputs/train "$PHASE1_DIR"

echo "=== Phase 1: chess positions (500k) ==="
uv run python src/sft_train.py \
    run_name="phase1-chess-500k" \
    data.train_files=data/sf_train_10m.jsonl \
    data.val_files=data/sf_val_10k.jsonl \
    data.train_max_samples=500000 \
    data.train_batch_size=512 \
    data.micro_batch_size_per_gpu=512 \
    trainer.default_local_dir="$PHASE1_DIR" \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    optim.lr=1e-3 \
    2>&1 | tee outputs/train/sft-phase1-$(date +%Y%m%d-%H%M%S).log

echo "=== Phase 2: English explanations ==="
# trainer.default_local_dir points at Phase 1's dir → auto-resumes from final checkpoint
uv run python src/sft_train.py \
    run_name="phase2-english" \
    data.train_files=data/english_50k.jsonl \
    data.val_files=data/sf_val_10k.jsonl \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=32 \
    trainer.default_local_dir="$PHASE1_DIR" \
    trainer.save_freq=10 \
    trainer.total_epochs=2 \
    optim.lr=1e-4 \
    2>&1 | tee outputs/train/sft-phase2-$(date +%Y%m%d-%H%M%S).log

echo "=== Two-phase SFT complete ==="
