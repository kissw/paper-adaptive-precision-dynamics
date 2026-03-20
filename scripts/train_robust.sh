#!/bin/bash
# Auto-restart training on CUDA errors. Resumes from latest checkpoint.

CONFIG="${1:-configs/default.yaml}"
DATA="${2:-data/expert_data.h5}"
OUTPUT_DIR="${3:-outputs/train_v1}"
MAX_RETRIES=20

CKPT_DIR="$OUTPUT_DIR/checkpoints"
LOG_FILE="$OUTPUT_DIR/train_robust.log"
mkdir -p "$CKPT_DIR"

for ((i=1; i<=MAX_RETRIES; i++)); do
    # Find latest epoch checkpoint
    LATEST=$(ls "$CKPT_DIR"/epoch_*.pt 2>/dev/null | sed 's/.*epoch_//' | sed 's/\.pt//' | sort -n | tail -1)
    [ -n "$LATEST" ] && LATEST="$CKPT_DIR/epoch_${LATEST}.pt"
    
    RESUME_ARGS=""
    if [ -n "$LATEST" ]; then
        RESUME_ARGS="--resume $LATEST"
        echo "[Attempt $i] Resuming from $LATEST" | tee -a "$LOG_FILE"
    else
        echo "[Attempt $i] Starting from scratch" | tee -a "$LOG_FILE"
    fi

    PYTHONUNBUFFERED=1 uv run python scripts/train.py \
        --config "$CONFIG" \
        --data "$DATA" \
        --output_dir "$OUTPUT_DIR" \
        $RESUME_ARGS \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Attempt $i] Training completed successfully!" | tee -a "$LOG_FILE"
        exit 0
    fi

    echo "[Attempt $i] Crashed with exit code $EXIT_CODE. Waiting 10s before restart..." | tee -a "$LOG_FILE"
    sleep 10
    
    # Reset CUDA state
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 5
done

echo "Exceeded $MAX_RETRIES retries. Giving up." | tee -a "$LOG_FILE"
exit 1
