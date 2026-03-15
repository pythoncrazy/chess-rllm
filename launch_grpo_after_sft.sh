#!/bin/bash
# Wait for SFT to complete, then launch GRPO from final checkpoint
# Usage: bash launch_grpo_after_sft.sh <sft_run_name> <grpo_run_name>
#   e.g. bash launch_grpo_after_sft.sh top3-cot-bs512 sft-warmstart-top3

SFT_RUN=${1:-top3-cot-bs512}
GRPO_RUN=${2:?Usage: $0 <sft_run_name> <grpo_run_name>}

LOG=/home/supersketchy/git/chess-rllm/outputs/sft-${SFT_RUN}.log
CKPT_DIR=/home/supersketchy/git/chess-rllm/checkpoints/sft/${SFT_RUN}/checkpoints.jsonl
GRPO_LOG=/home/supersketchy/git/chess-rllm/outputs/grpo-${GRPO_RUN}.log
GRPO_PIDFILE=/home/supersketchy/git/chess-rllm/outputs/grpo-${GRPO_RUN}.pid

echo "[$(date)] Monitoring SFT run for completion..." | tee -a $GRPO_LOG

while true; do
    if grep -q "Training completed successfully" "$LOG" 2>/dev/null; then
        echo "[$(date)] SFT training completed!" | tee -a $GRPO_LOG
        break
    fi
    sleep 60
done

# Get the final checkpoint path
FINAL_CKPT=$(grep '"final"' "$CKPT_DIR" 2>/dev/null | python3 -c "import sys, json; d = json.loads(sys.stdin.read().strip()); print(d['state_path'])" 2>/dev/null)

if [ -z "$FINAL_CKPT" ]; then
    # Fall back to last checkpoint
    FINAL_CKPT=$(tail -1 "$CKPT_DIR" | python3 -c "import sys, json; d = json.loads(sys.stdin.read().strip()); print(d['state_path'])")
fi

echo "[$(date)] Using checkpoint: $FINAL_CKPT" | tee -a $GRPO_LOG

# Clean up any stale GRPO checkpoints
rm -f /tmp/rllm-tinker-checkpoints/checkpoints.jsonl 2>/dev/null

cd /home/supersketchy/git/chess-rllm

echo "[$(date)] Launching GRPO training as run '${GRPO_RUN}'..." | tee -a $GRPO_LOG
nohup uv run python train.py "run_name=${GRPO_RUN}" "training.resume_from_tinker_id=${FINAL_CKPT}" >> $GRPO_LOG 2>&1 &
echo $! > $GRPO_PIDFILE
echo "[$(date)] GRPO launched with PID $(cat $GRPO_PIDFILE)" | tee -a $GRPO_LOG
