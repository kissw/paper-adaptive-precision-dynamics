"""Fit a Shared Token GMM contrastive preference model for TokenViT.

This script is the TOKEN-LEVEL counterpart of fit_contrastive_preference.py.
That script uses get_feat() -> pooled (B,320) for RSSM/direct-comparison paths.
This script uses concat(deter_token, stoch_token) -> (B,64,320) -> (B*64,320).

Key differences:
  fit_contrastive_preference.py  -> pooled GMM  -> 'contrastive_preference' key
  fit_token_contrastive_preference.py  -> Shared Token GMM -> 'token_contrastive_preference' key

Both keys can coexist in the same checkpoint.

Usage:
    ~/.local/bin/uv run python scripts/fit_token_contrastive_preference.py \\
        --checkpoint outputs/token_vit/checkpoints/best.pt \\
        --clean_data data/expert_data_town04.h5 \\
        --obstacle_data data/expert_data_v4.h5 \\
        --output outputs/token_vit/checkpoints/best_token_contrastive.pt \\
        --config configs/experiment/token_vit.yaml \\
        --K_clean 5 \\
        --K_avoid 5 \\
        --max_frames 5000 \\
        --max_token_samples 200000 \\
        --topk 1 4 8 16 \\
        --feature_type deter_stoch
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.training.token_preference import (
    TokenContrastivePreference,
    extract_token_features,
    topk_mean_token_score,
)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_token_latents(
    agent: DeepAIFAgent,
    data_path: str,
    max_frames: int,
    feature_type: str = "deter_stoch",
) -> torch.Tensor:
    """Encode HDF5 frames into per-token posterior features.

    Returns (n_frames, N, token_dim) on CPU.

    Raises ValueError if the world model is not TokenViT (i.e., state.deter
    is not 3D).  The check happens at the first obs_step call.
    """
    wm = agent.world_model
    dev = agent._device

    if wm._wm_type != "token_vit":
        raise ValueError(
            "Token-wise contrastive preference requires TokenViT state with 3D "
            "deter/stoch. Got world_model_type="
            f"{wm._wm_type!r}. "
            "Use fit_contrastive_preference.py for RSSM checkpoints."
        )

    with h5py.File(data_path, "r") as f:
        # Support both 'images' (training HDF5) and 'frames' (collection HDF5)
        img_key = "images" if "images" in f else "frames"
        n_avail = min(len(f[img_key]), max_frames)
        images = torch.tensor(f[img_key][:n_avail], dtype=torch.float32)
        states = torch.tensor(f["states"][:n_avail], dtype=torch.float32)

    all_tokens = []
    t0 = time.time()

    with torch.no_grad():
        state = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, agent._cfg.cem.action_dim, device=dev)

        for i in range(len(images)):
            img = images[i : i + 1].to(dev)
            st  = states[i : i + 1].to(dev)
            embed = wm.encoder(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)

            # extract_token_features validates 3D state at first call
            tok = extract_token_features(post, feature_type)  # (1, N, token_dim)
            all_tokens.append(tok.squeeze(0).cpu())            # (N, token_dim)

            state = type(post)(*[x.detach() for x in post])
            # prev_act left as zeros (no per-frame action conditioning)
            # consistent with fit_contrastive_preference.py convention

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(images)} frames  ({elapsed:.1f}s)")

    result = torch.stack(all_tokens)  # (n_frames, N, token_dim)
    print(f"  Encoded {result.shape[0]} frames × {result.shape[1]} tokens "
          f"× {result.shape[2]}d in {time.time()-t0:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Shared Token GMM contrastive preference for TokenViT",
    )
    # Required
    parser.add_argument("--checkpoint",    required=True, help="TokenViT checkpoint .pt")
    parser.add_argument("--clean_data",    required=True, help="HDF5 with obstacle-free frames")
    parser.add_argument("--obstacle_data", required=True, help="HDF5 with obstacle-present frames")
    parser.add_argument("--output",        required=True, help="Output checkpoint path")
    parser.add_argument("--config",        required=True, help="YAML config (must match checkpoint)")

    # GMM
    parser.add_argument("--K_clean",          type=int,   default=5)
    parser.add_argument("--K_avoid",          type=int,   default=5)
    parser.add_argument("--covariance_type",  default="diag",
                        help="diag (default). full not recommended for high-dim tokens.")
    parser.add_argument("--max_frames",       type=int,   default=5000,
                        help="Max frames to encode per dataset")
    parser.add_argument("--max_token_samples", type=int,  default=200000,
                        help="Max flattened token samples for GMM fitting")
    parser.add_argument("--fit_iters",        type=int,   default=300)
    parser.add_argument("--fit_lr",           type=float, default=0.01)
    parser.add_argument("--contrast_scale",   type=float, default=1.0)
    parser.add_argument("--min_std",          type=float, default=0.01)
    parser.add_argument("--seed",             type=int,   default=42)

    # Feature
    parser.add_argument("--feature_type",  default="deter_stoch",
                        choices=["deter_stoch", "deter_token_mean", "token_mean"],
                        help="Token feature construction. Default: deter_stoch")

    # Diagnostic aggregation
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Top-k values for diagnostic aggregation")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ----- Load agent -----
    print(f"Loading config from {args.config}")
    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    print(f"  world_model_type: {agent.world_model._wm_type}")

    if agent.world_model._wm_type != "token_vit":
        print(
            "ERROR: Token-wise contrastive preference requires TokenViT state "
            "with 3D deter/stoch.\n"
            f"  Got world_model_type={agent.world_model._wm_type!r}.\n"
            "  Use fit_contrastive_preference.py for RSSM checkpoints.",
            file=sys.stderr,
        )
        sys.exit(1)

    tv = cfg.token_vit
    feat_dims = {
        "deter_stoch":      tv.deter_dim + tv.stoch_dim,   # 320
        "deter_token_mean": tv.deter_dim + tv.stoch_dim,   # 320
        "token_mean":       tv.stoch_dim,                  # 64
    }
    token_dim = feat_dims[args.feature_type]
    print(f"  feature_type: {args.feature_type}  ->  token_dim={token_dim}")
    print(f"  num_tokens:   {tv.num_tokens}")

    # ----- Encode -----
    print(f"\nEncoding clean token features from {args.clean_data}...")
    clean_tokens = encode_token_latents(
        agent, args.clean_data, args.max_frames, args.feature_type,
    )  # (F_c, N, token_dim)

    print(f"\nEncoding obstacle token features from {args.obstacle_data}...")
    avoid_tokens = encode_token_latents(
        agent, args.obstacle_data, args.max_frames, args.feature_type,
    )  # (F_a, N, token_dim)

    # ----- Flatten -----
    clean_flat = clean_tokens.reshape(-1, token_dim)  # (F_c*N, token_dim)
    avoid_flat = avoid_tokens.reshape(-1, token_dim)  # (F_a*N, token_dim)

    print(f"\nFlattened token samples:")
    print(f"  clean: {clean_flat.shape[0]}  avoid: {avoid_flat.shape[0]}")

    # ----- Subsample if needed -----
    rng = torch.Generator().manual_seed(args.seed)
    if clean_flat.shape[0] > args.max_token_samples:
        idx = torch.randperm(clean_flat.shape[0], generator=rng)[:args.max_token_samples]
        clean_flat = clean_flat[idx]
        print(f"  Subsampled clean -> {clean_flat.shape[0]}")
    if avoid_flat.shape[0] > args.max_token_samples:
        idx = torch.randperm(avoid_flat.shape[0], generator=rng)[:args.max_token_samples]
        avoid_flat = avoid_flat[idx]
        print(f"  Subsampled avoid -> {avoid_flat.shape[0]}")

    # Move to device for GMM fitting
    dev = agent._device
    clean_flat = clean_flat.to(dev)
    avoid_flat = avoid_flat.to(dev)

    # ----- Fit -----
    print(f"\nFitting Shared Token GMM (K_clean={args.K_clean}, K_avoid={args.K_avoid})...")
    model = TokenContrastivePreference(
        K_clean=args.K_clean,
        K_avoid=args.K_avoid,
        token_dim=token_dim,
        min_std=args.min_std,
        contrast_scale=args.contrast_scale,
    ).to(dev)

    model.fit(clean_flat, avoid_flat, n_iters=args.fit_iters, lr=args.fit_lr)

    # ----- Top-k diagnostics per frame -----
    print(f"\nTop-k mean diagnostics (frame-level, using first 200 encoded frames):")
    with torch.no_grad():
        n_diag = min(200, clean_tokens.shape[0], avoid_tokens.shape[0])
        c_tok = clean_tokens[:n_diag].to(dev)  # (n_diag, N, token_dim)
        a_tok = avoid_tokens[:n_diag].to(dev)

        for k in sorted(set(args.topk)):
            c_scores = model.score_frames(c_tok, topk=k)  # (n_diag,)
            a_scores = model.score_frames(a_tok, topk=k)
            c_mean = c_scores.mean().item()
            a_mean = a_scores.mean().item()
            gap    = c_mean - a_mean
            print(
                f"  k={k:3d}:  clean={c_mean:+6.2f}  avoid={a_mean:+6.2f}  "
                f"gap={gap:+6.2f}  {'OK' if gap > 0 else 'WARN: gap <= 0'}"
            )

    # ----- Save -----
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)

    ckpt["token_contrastive_preference"] = {
        "type":          "shared_token_gmm",
        "feature_type":  args.feature_type,
        "token_dim":     token_dim,
        "num_tokens":    tv.num_tokens,
        "K_clean":       args.K_clean,
        "K_avoid":       args.K_avoid,
        "covariance_type": args.covariance_type,
        "contrast_scale": args.contrast_scale,
        "topk_default":  4,
        "topk_candidates": sorted(set(args.topk)),
        # GMM parameters (same format as pooled contrastive_preference)
        "clean_means":    model.clean.means.data.cpu(),
        "clean_log_stds": model.clean.log_stds.data.cpu(),
        "clean_logits":   model.clean.logits.data.cpu(),
        "avoid_means":    model.avoid.means.data.cpu(),
        "avoid_log_stds": model.avoid.log_stds.data.cpu(),
        "avoid_logits":   model.avoid.logits.data.cpu(),
        "diagnostics": {
            "clean_frames_encoded": clean_tokens.shape[0],
            "avoid_frames_encoded": avoid_tokens.shape[0],
            "clean_token_samples":  clean_flat.shape[0],
            "avoid_token_samples":  avoid_flat.shape[0],
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"\nSaved token contrastive preference to {args.output}")
    print(f"  Checkpoint keys: {list(ckpt.keys())}")
    print("  'contrastive_preference' (pooled) is unchanged if present.")
    print("  'token_contrastive_preference' (token GMM) is the new key.")


if __name__ == "__main__":
    main()
