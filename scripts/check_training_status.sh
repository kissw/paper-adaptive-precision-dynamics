#!/bin/bash
# Quick status check for 5D training
TRAIN_DIR="outputs/train_v7_5d"
echo "=== Training Status $(date) ==="

# Check if process is alive
if kill -0 952455 2>/dev/null; then
    echo "Training PID 952455: RUNNING"
else
    echo "Training PID 952455: FINISHED"
fi

# Check checkpoints
echo ""
echo "Checkpoints:"
ls -lh "$TRAIN_DIR/checkpoints/" 2>/dev/null || echo "  (none)"

# Check pipeline log
echo ""
echo "Pipeline log (last 5 lines):"
tail -5 "$TRAIN_DIR/pipeline_5d.log" 2>/dev/null || echo "  (none)"

# Check eval results if they exist
for d in outputs/eval_task_b_v4_5d outputs/eval_task_a_v7; do
    if [ -f "$d/eval_results.csv" ]; then
        echo ""
        echo "Eval results: $d"
        cat "$d/eval_results.csv"
    fi
done
