#!/bin/bash
# Task B: 5D State Pipeline (obstacle_distance as 5th state dimension)
#
# Steps:
# 1. Merge 5D Task B data with padded Task A data
# 2. Fine-tune world model with weight surgery (4D→5D)
# 3. Refit Task B preference (K=7) on lane-change data
# 4. Evaluate Task B (obstacle avoidance)
# 5. Evaluate Task A (regression check)

set -eo pipefail

TASK_B_DATA="data/task_b_lanechange_5d.h5"
TASK_A_DATA="data/expert_data_mixed.h5"
COMBINED_DATA="data/expert_data_v7_5d.h5"
TRAIN_DIR="outputs/train_v7_5d"
PREV_CHECKPOINT="outputs/train_v6_combined/checkpoints/best.pt"
EVAL_B_DIR="outputs/eval_task_b_v4_5d"
EVAL_A_DIR="outputs/eval_task_a_v7"

mkdir -p "$TRAIN_DIR" "$EVAL_B_DIR" "$EVAL_A_DIR"

echo "=============================================="
echo "Task B: 5D State Pipeline"
echo "=============================================="

# Step 1: Verify data exists
echo "[1/5] Checking data files..."
if [ ! -f "$TASK_B_DATA" ]; then
    echo "ERROR: Task B 5D data not found: $TASK_B_DATA"
    exit 1
fi
if [ ! -f "$TASK_A_DATA" ]; then
    echo "ERROR: Task A data not found: $TASK_A_DATA"
    exit 1
fi

# Step 2: Merge datasets (merge_data.py pads 4D→5D with 1.0)
echo "[2/5] Merging datasets (4D Task A + 5D Task B)..."
uv run python scripts/merge_data.py \
    "$TASK_A_DATA" "$TASK_B_DATA" \
    --output "$COMBINED_DATA" 2>&1 | tee "${TRAIN_DIR}/merge.log"

# Step 3: Fine-tune world model with 5D state
echo "[3/5] Fine-tuning world model (5D state, weight surgery from 4D checkpoint)..."
uv run python scripts/train.py \
    --config configs/experiment/task_b.yaml \
    --data "$COMBINED_DATA" \
    --epochs 20 \
    --resume "$PREV_CHECKPOINT" \
    --output_dir "$TRAIN_DIR" 2>&1 | tee "${TRAIN_DIR}/train_5d.log"

# Step 4: Refit Task B preference
echo "[4/5] Refitting Task B preference (K=7, lane-change only)..."
uv run python scripts/refit_preference.py \
    --checkpoint "${TRAIN_DIR}/checkpoints/best.pt" \
    --data "$COMBINED_DATA" \
    --config configs/experiment/task_b.yaml \
    --task_filter B --K 7 \
    --output "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" 2>&1 | tee "${TRAIN_DIR}/refit_5d.log"

# Step 5a: Evaluate Task B
echo "[5/5] Evaluating Task B (5D state, obstacle avoidance)..."
uv run python scripts/evaluate.py \
    --task B \
    --checkpoint "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" \
    --config configs/experiment/task_b.yaml \
    --num_obstacles 3 --episodes 3 --max_frames 3000 \
    --save_video --output_dir "$EVAL_B_DIR" 2>&1 | tee "${EVAL_B_DIR}/eval_task_b_5d.log"

# Step 5b: Task A regression check
echo "Evaluating Task A regression (5D state, no obstacles)..."
uv run python scripts/evaluate.py \
    --task A \
    --checkpoint "${TRAIN_DIR}/checkpoints/best_taskb_pref_5d.pt" \
    --config configs/experiment/task_b.yaml \
    --episodes 3 --max_frames 3000 \
    --save_video --output_dir "$EVAL_A_DIR" 2>&1 | tee "${EVAL_A_DIR}/eval_task_a_v7.log"

echo "=============================================="
echo "Task B 5D Pipeline Complete"
echo "=============================================="
echo ""
echo "Results:"
echo "  Task B eval: ${EVAL_B_DIR}/eval_results.csv"
echo "  Task B videos: ${EVAL_B_DIR}/*.mp4"
echo "  Task A regression: ${EVAL_A_DIR}/eval_results.csv"
echo "  Training log: ${TRAIN_DIR}/train_5d.log"
