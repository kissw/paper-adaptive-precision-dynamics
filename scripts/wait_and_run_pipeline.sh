#!/bin/bash
# Wait for training PID to finish, then run the Task B pipeline
set -eo pipefail

TRAIN_PID=$1
echo "Waiting for training process (PID $TRAIN_PID) to complete..."

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    # Check latest checkpoint for progress
    LATEST=$(ls -t outputs/train_v6_combined/checkpoints/epoch_*.pt 2>/dev/null | head -1)
    echo "  $(date '+%H:%M:%S') Training still running... Latest: ${LATEST:-none}"
    sleep 300
done

echo ""
echo "Training process completed at $(date)"
echo "Checkpoints:"
ls -lh outputs/train_v6_combined/checkpoints/

echo ""
echo "Starting Task B pipeline..."
bash scripts/run_task_b_pipeline.sh
