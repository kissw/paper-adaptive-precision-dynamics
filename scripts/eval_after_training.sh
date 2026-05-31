#!/usr/bin/env bash
set -e

cd ~/av/paper-adaptive-precision-dynamics

# ============================================================
# 0. Latest corrected runs
# ============================================================
RSSM_RUN=$(ls -td runs/smoke/rssm_64_cropfix_* | head -n 1)
VIT_RUN=$(ls -td runs/smoke/token_vit_64_cropfix_* | head -n 1)

RSSM_CKPT="$RSSM_RUN/checkpoints/best.pt"
VIT_CKPT="$VIT_RUN/checkpoints/best.pt"

# RSSM_FINAL=/tmp/rssm_64_cropfix_best.pt
# VIT_FINAL=/tmp/token_vit_64_cropfix_best.pt
RSSM_FINAL=$RSSM_CKPT
VIT_FINAL=$VIT_CKPT

# cp "$RSSM_CKPT" "$RSSM_FINAL"
# cp "$VIT_CKPT" "$VIT_FINAL"

EVAL_DIR="runs/smoke/eval_cropfix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"

echo "RSSM_RUN=$RSSM_RUN"
echo "VIT_RUN=$VIT_RUN"
echo "RSSM_FINAL=$RSSM_FINAL"
echo "VIT_FINAL=$VIT_FINAL"
echo "EVAL_DIR=$EVAL_DIR"


# ============================================================
# 1. Check checkpoint metadata
# ============================================================
~/.local/bin/uv run python - <<PY | tee "$EVAL_DIR/checkpoint_metadata.txt"
import torch

for name, path in [
    ("RSSM", "$RSSM_FINAL"),
    ("TokenViT", "$VIT_FINAL"),
]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("=" * 100)
    print(name, path)
    print("top-level keys:", list(ckpt.keys()))
    for k in [
        "epoch", "global_step", "best_epoch", "best_loss", "train_loss",
        "checkpoint_type", "is_best", "world_model_type", "crop_road",
        "image_size", "data_path", "output_dir", "git_branch", "git_commit",
        "timestamp"
    ]:
        print(k, "=", ckpt.get(k, None))
PY


# ============================================================
# 2. Verify corrected crop-target reconstruction behavior
#    Expected after retraining:
#    MSE(recon, preprocessed/crop image) < MSE(recon, full image)
# ============================================================
~/.local/bin/uv run python - <<'PY' | tee "$EVAL_DIR/reconstruction_target_check.txt"
import h5py
import torch
import torch.nn.functional as F

from active_inference.config import Config
from active_inference.agent import WorldModel

cases = [
    ("RSSM", "configs/experiment/task_b_v5.yaml", "runs/smoke/rssm_64_cropfix_20260530_063140/checkpoints/best.pt"),
    ("TokenViT", "configs/experiment/token_vit.yaml", "runs/smoke/token_vit_64_cropfix_20260530_063159/checkpoints/best.pt"),
]

data_path = "data/expert_data_v4.h5"
idx = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with h5py.File(data_path, "r") as f:
    img_full = torch.tensor(f["images"][idx:idx+1], dtype=torch.float32).to(device)
    state_vec = torch.tensor(f["states"][idx:idx+1], dtype=torch.float32).to(device)
    action_dim = f["actions"].shape[-1]

for name, cfg_path, ckpt_path in cases:
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"], strict=True)
    wm.eval()

    action = torch.zeros(1, action_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        img_model = wm.preprocess_image(img_full)
        rssm_state = wm.rssm.initial(1, device)
        embed = wm.encode_obs(img_full, state_vec)
        post, prior = wm.rssm.obs_step(rssm_state, action, embed)
        recon = wm.decode_obs(post)

    mse_full = F.mse_loss(recon, img_full).item()
    mse_model = F.mse_loss(recon, img_model).item()

    print("=" * 100)
    print(name)
    print("cfg:", cfg_path)
    print("ckpt:", ckpt_path)
    print("cfg.encoder.crop_road:", cfg.encoder.crop_road)
    print("wm._crop_road:", getattr(wm, "_crop_road", None))
    print("wm.encoder._crop_road:", getattr(wm.encoder, "_crop_road", None))
    print("img_full shape:", tuple(img_full.shape), "range:", float(img_full.min()), float(img_full.max()))
    print("img_model shape:", tuple(img_model.shape), "range:", float(img_model.min()), float(img_model.max()))
    print("recon shape:", tuple(recon.shape), "range:", float(recon.min()), float(recon.max()))
    print("MSE(recon, full image):", mse_full)
    print("MSE(recon, preprocessed/crop image):", mse_model)
    if mse_model < mse_full:
        print("Result: OK. Recon is closer to crop/preprocessed target.")
    else:
        print("Result: WARNING. Recon is still closer to full image.")
    print("=" * 100)
PY


# ============================================================
# 3. RSSM pooled contrastive GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_contrastive_preference.py \
    --checkpoint "$RSSM_FINAL" \
    --clean_data data/expert_data_town04.h5 \
    --obstacle_data data/expert_data_v4.h5 \
    --output "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.pt" \
    --config configs/experiment/task_b_v5.yaml \
    --K_clean 5 \
    --K_avoid 7 \
    --max_samples 20000 \
    2>&1 | tee "$EVAL_DIR/rssm_pooled_contrastive_k5k7_s20000.log"


# ============================================================
# 4. TokenViT pooled contrastive GMM
# ============================================================
~/.local/bin/uv run python scripts/fit_contrastive_preference.py \
    --checkpoint "$VIT_FINAL" \
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
    --checkpoint "$VIT_FINAL" \
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
    --checkpoint "$VIT_FINAL" \
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
    --checkpoint "$VIT_FINAL" \
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


# ============================================================
# 8. RSSM vs TokenViT rollout visualization, obstacle H=15
# ============================================================
~/.local/bin/uv run python scripts/compare_rssm_token_vit_rollout.py \
    --rssm_checkpoint "$RSSM_FINAL" \
    --rssm_config configs/experiment/task_b_v5.yaml \
    --token_vit_checkpoint "$VIT_FINAL" \
    --token_vit_config configs/experiment/token_vit.yaml \
    --data data/expert_data_v4.h5 \
    --start_index 1000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 10 \
    --stride 1000 \
    --output_dir "$EVAL_DIR/rollout_obstacle_h15" \
    --case_name obstacle_h15 \
    --device cuda \
    2>&1 | tee "$EVAL_DIR/rollout_obstacle_h15.log"


# ============================================================
# 9. RSSM vs TokenViT rollout visualization, clean H=15
# ============================================================
~/.local/bin/uv run python scripts/compare_rssm_token_vit_rollout.py \
    --rssm_checkpoint "$RSSM_FINAL" \
    --rssm_config configs/experiment/task_b_v5.yaml \
    --token_vit_checkpoint "$VIT_FINAL" \
    --token_vit_config configs/experiment/token_vit.yaml \
    --data data/expert_data_town04.h5 \
    --start_index 1000 \
    --context_len 5 \
    --horizon 15 \
    --num_cases 10 \
    --stride 1000 \
    --output_dir "$EVAL_DIR/rollout_clean_h15" \
    --case_name clean_h15 \
    --device cuda \
    2>&1 | tee "$EVAL_DIR/rollout_clean_h15.log"


# ============================================================
# 10. Rollout metrics summary
# ============================================================
~/.local/bin/uv run python - <<PY | tee "$EVAL_DIR/rollout_metrics_summary.txt"
import pandas as pd
from pathlib import Path

for d in [
    Path("$EVAL_DIR/rollout_obstacle_h15"),
    Path("$EVAL_DIR/rollout_clean_h15"),
]:
    print("=" * 100)
    print(d)
    csvs = list(d.glob("*metrics*.csv"))
    if not csvs:
        print("No metrics CSV found")
        continue

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    print("\nMean by model:")
    print(df.groupby("model")[["mse", "psnr", "ssim"]].mean().round(4))

    print("\nMean by model/horizon:")
    print(df.groupby(["model", "horizon_step"])[["mse", "psnr", "ssim"]].mean().round(4))
PY


# ============================================================
# 11. Quick result file listing
# ============================================================
echo "Evaluation complete."
echo "EVAL_DIR=$EVAL_DIR"
find "$EVAL_DIR" -maxdepth 3 -type f | sort | tee "$EVAL_DIR/file_list.txt"