#!/usr/bin/env bash
set -e

cd ~/av/paper-adaptive-precision-dynamics

# ============================================================
# 0. Latest corrected runs
# ============================================================
RSSM_RUN=$(ls -td runs/rssm_64_cropfix_* | head -n 1)
VIT_RUN=$(ls -td runs/token_vit_64_cropfix_* | head -n 1)

RSSM_CKPT="$RSSM_RUN/checkpoints/best.pt"
VIT_CKPT="$VIT_RUN/checkpoints/best.pt"

EVAL_DIR="runs/eval_cropfix_20260601_201346"
# mkdir -p "$EVAL_DIR"

echo "RSSM_RUN=$RSSM_RUN"
echo "VIT_RUN=$VIT_RUN"
echo "EVAL_DIR=$EVAL_DIR"


~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/default.yaml \
    --preference_type pooled_contrastive \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 22000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 20 \
    --stride 500 \
    --output_dir "$EVAL_DIR/pp_gap_rssm_pooled"

~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/tokenvit_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/experiment/token_vit.yaml \
    --preference_type pooled_contrastive \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 22000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 20 \
    --stride 500 \
    --output_dir "$EVAL_DIR/pp_gap_tokenvit_pooled"

~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/tokenvit_shared_token_contrastive_k5k7_f20000_t200k.pt" \
    --config configs/experiment/token_vit.yaml \
    --preference_type token_contrastive \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 22000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 20 \
    --stride 500 \
    --topk 4 \
    --output_dir "$EVAL_DIR/pp_gap_tokenvit_shared_topk4"

~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/tokenvit_posnorm_token_contrastive_k5k7_f20000_t200k.pt" \
    --config configs/experiment/token_vit.yaml \
    --preference_type token_contrastive \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 22000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 20 \
    --stride 500 \
    --topk 4 \
    --output_dir "$EVAL_DIR/pp_gap_tokenvit_posnorm_topk4"

~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/tokenvit_vampprior_q010_p005_f20000.pt" \
    --config configs/experiment/token_vit.yaml \
    --preference_type token_vampprior \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 22000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 20 \
    --stride 500 \
    --topk 4 \
    --output_dir "$EVAL_DIR/pp_gap_tokenvit_vampprior_topk4" 

# ============================================================
# 11. Quick result file listing
# ============================================================
echo "Evaluation complete."
echo "EVAL_DIR=$EVAL_DIR"
find "$EVAL_DIR" -maxdepth 3 -type f | sort | tee "$EVAL_DIR/file_list.txt"