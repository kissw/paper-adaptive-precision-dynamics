#!/bin/bash
# Run all three evaluation tasks sequentially after training
# Usage: bash scripts/run_evaluation.sh [checkpoint_path]

set -e

CKPT="${1:-outputs/train_v4/checkpoints/best.pt}"
CONFIG="configs/default.yaml"
EVAL_BASE="outputs/eval_v4"
EPISODES=3
MAX_FRAMES=2000

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

echo "============================================"
echo "Evaluation Suite — Deep AIF Agent v4"
echo "Checkpoint: $CKPT"
echo "Episodes per route: $EPISODES"
echo "============================================"

# Task A: Highway curves (Town04)
echo ""
echo "[1/3] Task A — Highway Curves (Town04)"
echo "--------------------------------------------"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate.py \
    --task A \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --max_frames "$MAX_FRAMES" \
    --save_video \
    --output_dir "$EVAL_BASE/task_a"

# Baseline: Highway straight (Town06_Opt)
echo ""
echo "[2/3] Baseline — Highway Straight (Town06_Opt)"
echo "--------------------------------------------"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate.py \
    --task baseline \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --max_frames "$MAX_FRAMES" \
    --save_video \
    --output_dir "$EVAL_BASE/baseline"

# Task B: Obstacle avoidance (Town06_Opt)
echo ""
echo "[3/3] Task B — Obstacle Avoidance (Town06_Opt)"
echo "--------------------------------------------"
PYTHONUNBUFFERED=1 uv run python scripts/evaluate.py \
    --task B \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --max_frames "$MAX_FRAMES" \
    --save_video \
    --num_obstacles 3 \
    --output_dir "$EVAL_BASE/task_b"

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "Results saved to: $EVAL_BASE/"
echo "============================================"
