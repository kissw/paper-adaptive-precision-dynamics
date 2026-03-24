#!/bin/bash
# Task B Pipeline: Runs after training completes
# Stages: Refit Task B preference -> Evaluate Task B -> Generate results
set -eo pipefail

TRAIN_DIR="outputs/train_v6_combined"
CHECKPOINT="${TRAIN_DIR}/checkpoints/best.pt"
DATA="data/expert_data_v6_combined.h5"
EVAL_DIR="outputs/eval_task_b_v1"
EVAL_A_DIR="outputs/eval_task_a_v6"
TASKB_PREF="${TRAIN_DIR}/checkpoints/best_taskb_pref.pt"

# Create output directories upfront
mkdir -p "$EVAL_DIR" "$EVAL_A_DIR" "${TRAIN_DIR}"

echo "=============================================="
echo "Task B Pipeline"
echo "=============================================="

# Wait for training checkpoint to exist
echo "[1/4] Waiting for training checkpoint..."
while [ ! -f "$CHECKPOINT" ]; do
    echo "  Waiting for ${CHECKPOINT}..."
    sleep 300
done
echo "  Checkpoint found: ${CHECKPOINT}"

# Stage 1: Refit Task B preference (lane-change episodes only)
echo ""
echo "[2/4] Refitting Task B preference model (K=7, lane-change only)..."
uv run python scripts/refit_preference.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --config configs/experiment/task_b.yaml \
    --output "$TASKB_PREF" \
    --task_filter B \
    --K 7 \
    --min_speed 1.0 \
    --max_samples 5000 \
    --fit_iters 300 \
    --fit_lr 0.005 \
    2>&1 | tee "${TRAIN_DIR}/refit_taskb.log"

echo ""
echo "  Task B preference saved: ${TASKB_PREF}"

# Stage 2: Evaluate Task B with videos
echo ""
echo "[3/4] Evaluating Task B (3 episodes, 3 obstacles, with video)..."
uv run python scripts/evaluate.py \
    --task B \
    --checkpoint "$TASKB_PREF" \
    --config configs/experiment/task_b.yaml \
    --num_obstacles 3 \
    --episodes 3 \
    --max_frames 3000 \
    --save_video \
    --output_dir "$EVAL_DIR" \
    2>&1 | tee "${EVAL_DIR}/eval_task_b.log"

echo ""
echo "  Evaluation results: ${EVAL_DIR}/eval_results.csv"
echo "  Videos: ${EVAL_DIR}/*.mp4"

# Stage 3: Also evaluate Task A with the combined world model
# to ensure no regression
echo ""
echo "[4/4] Evaluating Task A (regression check, no video)..."
uv run python scripts/evaluate.py \
    --task A \
    --checkpoint "$CHECKPOINT" \
    --config configs/default.yaml \
    --episodes 3 \
    --max_frames 3000 \
    --save_video \
    --output_dir "$EVAL_A_DIR" \
    --route_index 2 \
    2>&1 | tee "${EVAL_A_DIR}/eval_task_a.log"

echo ""
echo "=============================================="
echo "Task B Pipeline Complete"
echo "=============================================="
echo ""
echo "Results:"
echo "  Task B eval: ${EVAL_DIR}/eval_results.csv"
echo "  Task B videos: ${EVAL_DIR}/*.mp4"
echo "  Task A regression: ${EVAL_A_DIR}/eval_results.csv"
echo "  Training log: ${TRAIN_DIR}/train_v6_combined.log"
