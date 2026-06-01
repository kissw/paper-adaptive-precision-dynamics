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

EVAL_DIR="runs/eval_cropfix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"

echo "RSSM_RUN=$RSSM_RUN"
echo "VIT_RUN=$VIT_RUN"
echo "EVAL_DIR=$EVAL_DIR"


# ============================================================
# 3. RSSM pooled contrastive GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_contrastive_preference.py \
    --checkpoint "$RSSM_CKPT" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/default.yaml \
    --K_clean 5 \
    --K_avoid 7 \
    --max_samples 20000 \
    2>&1 | tee "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.log"


# ============================================================
# 4. TokenViT pooled contrastive GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_contrastive_preference.py \
    --checkpoint "$VIT_CKPT" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/tokenvit_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/experiment/token_vit.yaml \
    --K_clean 5 \
    --K_avoid 7 \
    --max_samples 20000 \
    2>&1 | tee "$EVAL_DIR/tokenvit_pooled_contrastive_k5k7_s20000.log"


# ============================================================
# 5. TokenViT Shared Token GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_token_contrastive_preference.py \
    --checkpoint "$VIT_CKPT" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/tokenvit_shared_token_contrastive_k5k7_f20000_t200k.pt" \
    --config configs/experiment/token_vit.yaml \
    --mode shared \
    --K_clean 5 \
    --K_avoid 7 \
    --covariance_type diag \
    --max_frames 20000 \
    --max_token_samples 200000 \
    --topk 1 4 8 16 \
    --topk_default 4 \
    --feature_type deter_stoch \
    2>&1 | tee "$EVAL_DIR/tokenvit_shared_token_contrastive_k5k7_f20000_t200k.log"


# ============================================================
# 6. TokenViT Position-normalized Shared Token GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_token_contrastive_preference.py \
    --checkpoint "$VIT_CKPT" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/tokenvit_posnorm_token_contrastive_k5k7_f20000_t200k.pt" \
    --config configs/experiment/token_vit.yaml \
    --mode pos_norm \
    --K_clean 5 \
    --K_avoid 7 \
    --covariance_type diag \
    --max_frames 20000 \
    --max_token_samples 200000 \
    --topk 1 4 8 16 \
    --topk_default 4 \
    --feature_type deter_stoch \
    2>&1 | tee "$EVAL_DIR/tokenvit_posnorm_token_contrastive_k5k7_f20000_t200k.log"


# ============================================================
# 7. TokenViT VampPrior-like, strongest previous candidate
#    q_std_scale=0.10, proto_std_scale=0.05
# ============================================================
~/.local/bin/uv run python scripts/fit_token_vampprior_preference.py \
    --checkpoint "$VIT_CKPT" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/tokenvit_vampprior_q010_p005_f20000.pt" \
    --config configs/experiment/token_vit.yaml \
    --K_clean 5 \
    --K_avoid 7 \
    --max_frames 20000 \
    --min_std 0.01 \
    --contrast_scale 1.0 \
    --topk 1 4 8 16 \
    --topk_default 4 \
    --score_mode q_integrated \
    --q_std_scale 0.10 \
    --proto_std_scale 0.05 \
    2>&1 | tee "$EVAL_DIR/tokenvit_vampprior_q010_p005_f20000.log"


~/.local/bin/uv run python scripts/evaluate_posterior_prior_gap.py \
    --checkpoint "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/default.yaml \
    --preference_type pooled_contrastive \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --start_index 1000 \
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
    --start_index 1000 \
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
    --start_index 1000 \
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
    --start_index 1000 \
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
    --start_index 1000 \
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