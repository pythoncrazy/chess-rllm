#!/bin/bash
# Wait for SFT to complete, then launch GRPO from final checkpoint

LOG=/home/supersketchy/git/chess-rllm/outputs/sft-top3-cot-bs512.log
CKPT_DIR=/home/supersketchy/git/chess-rllm/checkpoints/sft/top3-cot-bs512/checkpoints.jsonl
GRPO_LOG=/home/supersketchy/git/chess-rllm/outputs/grpo-from-sft.log
GRPO_PIDFILE=/home/supersketchy/git/chess-rllm/outputs/grpo.pid

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

echo "[$(date)] Launching GRPO training..." | tee -a $GRPO_LOG
nohup uv run python train.py "training.resume_from_tinker_id=${FINAL_CKPT}" >> $GRPO_LOG 2>&1 &
echo $! > $GRPO_PIDFILE
echo "[$(date)] GRPO launched with PID $(cat $GRPO_PIDFILE)" | tee -a $GRPO_LOG
