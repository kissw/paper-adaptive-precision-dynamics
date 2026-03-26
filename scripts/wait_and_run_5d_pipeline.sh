#!/bin/bash
# Wait for 5D training to complete, then run refit + eval pipeline
set -eo pipefail

TRAIN_PID="$1"
if [ -z "$TRAIN_PID" ]; then
    echo "Usage: $0 <training_PID>"
    exit 1
fi

TRAIN_DIR="outputs/train_v7_5d"
COMBINED_DATA="data/expert_data_v7_5d.h5"
EVAL_B_DIR="outputs/eval_task_b_v4_5d"
EVAL_A_DIR="outputs/eval_task_a_v7"
CONFIG="configs/experiment/task_b.yaml"

mkdir -p "$EVAL_B_DIR" "$EVAL_A_DIR"

echo "Waiting for training PID $TRAIN_PID to finish..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 300
    echo "  $(date +%H:%M) - Training still running..."
done
echo "Training complete at $(date)"

# Check training succeeded
if [ ! -f "${TRAIN_DIR}/checkpoints/best.pt" ]; then
    echo "ERROR: No best.pt checkpoint found. Training may have failed."
    exit 1
fi

echo ""
echo "=============================================="
echo "Post-training: Refit + Evaluate"
echo "=============================================="

# Step 1: Refit Task B preference (K=7, lane-change only)
echo "[1/3] Refitting Task B preference (K=7, lane-change only)..."
uv run python scripts/refit_preference.py \
    --checkpoint "${TRAIN_DIR}/checkpoints/best.pt" \
    --data "$COMBINED_DATA" \
    --config "$CONFIG" \
    --task_filter B --K 7 \
    --output "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" 2>&1 | tee "${TRAIN_DIR}/refit_5d.log"

echo "  Task B preference saved: ${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt"

# Step 2: Evaluate Task B (obstacle avoidance)
echo "[2/3] Evaluating Task B (5D state, obstacle avoidance)..."
uv run python scripts/evaluate.py \
    --task B \
    --checkpoint "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" \
    --config "$CONFIG" \
    --num_obstacles 3 --episodes 3 --max_frames 3000 \
    --save_video --output_dir "$EVAL_B_DIR" 2>&1 | tee "${EVAL_B_DIR}/eval_task_b_5d.log"

# Step 3: Task A regression check
echo "[3/3] Evaluating Task A regression (5D state, no obstacles)..."
uv run python scripts/evaluate.py \
    --task A \
    --checkpoint "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" \
    --config "$CONFIG" \
    --episodes 3 --max_frames 3000 \
    --save_video --output_dir "$EVAL_A_DIR" 2>&1 | tee "${EVAL_A_DIR}/eval_task_a_v7.log"

echo ""
echo "=============================================="
echo "5D Pipeline Complete"
echo "=============================================="
echo "Results:"
echo "  Task B eval: ${EVAL_B_DIR}/eval_results.csv"
echo "  Task B videos: ${EVAL_B_DIR}/*.mp4"
echo "  Task A regression: ${EVAL_A_DIR}/eval_results.csv"
echo "  Training log: ${TRAIN_DIR}/train_5d.log"
