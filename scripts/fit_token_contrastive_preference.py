"""Fit a Shared or Position-normalised Token GMM contrastive preference for TokenViT.

This script is the TOKEN-LEVEL counterpart of fit_contrastive_preference.py.
That script uses get_feat() -> pooled (B,320) for RSSM/direct-comparison.
This script uses concat(deter, stoch) -> (B,64,320) -> (B*64,320).

Supported modes
---------------
  shared    No position correction. All 64 token features from all frames
            are treated as independent samples from a shared distribution.

  pos_norm  Position-normalised. Per-position mean/std computed from clean
            tokens only; same statistics applied to both clean and obstacle
            tokens before GMM fitting. Reduces spurious inter-position
            variance caused by spatial position bias in the encoder.
            Default and recommended mode.

Key identifiers
---------------
  fit_contrastive_preference.py  -> pooled GMM  -> 'contrastive_preference'
  fit_token_contrastive_preference.py -> token GMM -> 'token_contrastive_preference'

Both keys can coexist in the same checkpoint.

Usage (pos_norm, recommended)
    ~/.local/bin/uv run python scripts/fit_token_contrastive_preference.py \\
        --checkpoint outputs/token_vit/best.pt \\
        --clean_data data/expert_data_town04.h5 \\
        --obstacle_data data/expert_data_v4.h5 \\
        --output outputs/token_vit/best_token_pref.pt \\
        --config configs/experiment/token_vit.yaml \\
        --mode pos_norm \\
        --K_clean 5 --K_avoid 5 \\
        --max_frames 5000 --max_token_samples 200000 \\
        --topk 1 4 8 16 --topk_default 4
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
    compute_position_stats,
    apply_position_normalization,
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
    """
    wm  = agent.world_model
    dev = agent._device

    with h5py.File(data_path, "r") as f:
        img_key = "images" if "images" in f else "frames"
        n_avail = min(len(f[img_key]), max_frames)
        images = torch.tensor(f[img_key][:n_avail], dtype=torch.float32)
        states = torch.tensor(f["states"][:n_avail], dtype=torch.float32)

    all_tokens = []
    t0 = time.time()

    with torch.no_grad():
        state    = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, agent._cfg.cem.action_dim, device=dev)

        for i in range(len(images)):
            img   = images[i : i + 1].to(dev)
            st    = states[i : i + 1].to(dev)
            embed = wm.encoder(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)

            tok = extract_token_features(post, feature_type)  # (1, N, D)
            all_tokens.append(tok.squeeze(0).cpu())            # (N, D)

            state = type(post)(*[x.detach() for x in post])

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(images)} frames  ({time.time()-t0:.1f}s)")

    result = torch.stack(all_tokens)          # (n_frames, N, D)
    print(
        f"  Encoded {result.shape[0]} frames × "
        f"{result.shape[1]} tokens × {result.shape[2]}d  "
        f"in {time.time()-t0:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Shared / Position-normalised Token GMM for TokenViT",
    )

    # Required
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--clean_data",    required=True)
    parser.add_argument("--obstacle_data", required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--config",        required=True)

    # Mode
    parser.add_argument(
        "--mode", default="pos_norm", choices=["shared", "pos_norm"],
        help="shared: no position correction (baseline). "
             "pos_norm: normalise per spatial position using clean stats (default/proposed).",
    )

    # GMM
    parser.add_argument("--K_clean",           type=int,   default=5)
    parser.add_argument("--K_avoid",           type=int,   default=5)
    parser.add_argument("--covariance_type",   default="diag")
    parser.add_argument("--max_frames",        type=int,   default=5000)
    parser.add_argument("--max_token_samples", type=int,   default=200000)
    parser.add_argument("--fit_iters",         type=int,   default=300)
    parser.add_argument("--fit_lr",            type=float, default=0.01)
    parser.add_argument("--contrast_scale",    type=float, default=1.0)
    parser.add_argument("--min_std",           type=float, default=0.01)
    parser.add_argument("--seed",              type=int,   default=42)

    # Feature type
    parser.add_argument(
        "--feature_type", default="deter_stoch",
        choices=["deter_stoch", "deter_token_mean", "token_mean"],
    )

    # Aggregation diagnostic
    parser.add_argument("--topk",         type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--topk_default", type=int, default=4)

    # Position normalisation
    parser.add_argument("--pos_norm_eps", type=float, default=1e-6)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # ----- Load agent -----
    print(f"Loading config: {args.config}")
    cfg   = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

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
        "deter_stoch":      tv.deter_dim + tv.stoch_dim,
        "deter_token_mean": tv.deter_dim + tv.stoch_dim,
        "token_mean":       tv.stoch_dim,
    }
    token_dim   = feat_dims[args.feature_type]
    num_tokens  = tv.num_tokens

    print(f"  world_model_type : {agent.world_model._wm_type}")
    print(f"  mode             : {args.mode}")
    print(f"  feature_type     : {args.feature_type}  ->  token_dim={token_dim}")
    print(f"  num_tokens       : {num_tokens}")

    # ----- Encode raw token features -----
    print(f"\nEncoding clean tokens from {args.clean_data}...")
    clean_tokens = encode_token_latents(
        agent, args.clean_data, args.max_frames, args.feature_type,
    )  # (F_c, N, D)

    print(f"\nEncoding obstacle tokens from {args.obstacle_data}...")
    avoid_tokens = encode_token_latents(
        agent, args.obstacle_data, args.max_frames, args.feature_type,
    )  # (F_a, N, D)

    # ----- Position normalisation (pos_norm mode) -----
    pos_stats_saved: dict | None = None

    if args.mode == "pos_norm":
        print(
            f"\nComputing position stats from clean tokens "
            f"(eps={args.pos_norm_eps})..."
        )
        mu_pos, std_pos = compute_position_stats(
            clean_tokens, eps=args.pos_norm_eps,
        )
        print(f"  mu_pos:  {tuple(mu_pos.shape)}  "
              f"mean_abs={mu_pos.abs().mean():.4f}")
        print(f"  std_pos: {tuple(std_pos.shape)}  "
              f"mean={std_pos.mean():.4f}  min={std_pos.min():.4f}")

        clean_tokens_proc = apply_position_normalization(
            clean_tokens, mu_pos, std_pos,
        )
        avoid_tokens_proc = apply_position_normalization(
            avoid_tokens, mu_pos, std_pos,
        )
        pos_stats_saved = {
            "mu_pos":  mu_pos.cpu(),
            "std_pos": std_pos.cpu(),
            "eps":     args.pos_norm_eps,
        }
        print("  Normalisation applied to both clean and obstacle tokens.")
    else:
        clean_tokens_proc = clean_tokens
        avoid_tokens_proc = avoid_tokens

    # ----- Flatten -----
    clean_flat = clean_tokens_proc.reshape(-1, token_dim)  # (F_c*N, D)
    avoid_flat = avoid_tokens_proc.reshape(-1, token_dim)  # (F_a*N, D)

    print(f"\nFlattened token samples:")
    print(f"  clean: {clean_flat.shape[0]}   avoid: {avoid_flat.shape[0]}")

    # ----- Subsample -----
    rng = torch.Generator().manual_seed(args.seed)
    if clean_flat.shape[0] > args.max_token_samples:
        idx = torch.randperm(clean_flat.shape[0], generator=rng)[: args.max_token_samples]
        clean_flat = clean_flat[idx]
        print(f"  Subsampled clean -> {clean_flat.shape[0]}")
    if avoid_flat.shape[0] > args.max_token_samples:
        idx = torch.randperm(avoid_flat.shape[0], generator=rng)[: args.max_token_samples]
        avoid_flat = avoid_flat[idx]
        print(f"  Subsampled avoid -> {avoid_flat.shape[0]}")

    dev = agent._device
    clean_flat = clean_flat.to(dev)
    avoid_flat = avoid_flat.to(dev)

    # ----- Fit GMM -----
    mode_label = "Position-normalised" if args.mode == "pos_norm" else "Shared"
    print(
        f"\nFitting {mode_label} Token GMM "
        f"(K_clean={args.K_clean}, K_avoid={args.K_avoid})..."
    )
    model = TokenContrastivePreference(
        K_clean=args.K_clean,
        K_avoid=args.K_avoid,
        token_dim=token_dim,
        min_std=args.min_std,
        contrast_scale=args.contrast_scale,
    ).to(dev)
    model.fit(clean_flat, avoid_flat, n_iters=args.fit_iters, lr=args.fit_lr)

    # ----- Top-k diagnostics (frame level, processed tokens) -----
    print(f"\nTop-k mean diagnostics (frame-level, mode={args.mode}):")
    with torch.no_grad():
        n_diag = min(200, clean_tokens_proc.shape[0], avoid_tokens_proc.shape[0])
        c_tok  = clean_tokens_proc[:n_diag].to(dev)
        a_tok  = avoid_tokens_proc[:n_diag].to(dev)

        for k in sorted(set(args.topk)):
            c_scores = model.score_frames(c_tok, topk=k)
            a_scores = model.score_frames(a_tok, topk=k)
            c_mean   = c_scores.mean().item()
            a_mean   = a_scores.mean().item()
            gap      = c_mean - a_mean
            status   = "OK" if gap > 0 else "WARN: gap <= 0"
            print(
                f"  k={k:3d}:  clean={c_mean:+7.2f}  avoid={a_mean:+7.2f}  "
                f"gap={gap:+7.2f}  {status}"
            )

    # ----- Save -----
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)

    tcp: dict = {
        "type":            "shared_token_gmm",
        "mode":            args.mode,
        "feature_type":    args.feature_type,
        "token_dim":       token_dim,
        "num_tokens":      num_tokens,
        "K_clean":         args.K_clean,
        "K_avoid":         args.K_avoid,
        "covariance_type": args.covariance_type,
        "contrast_scale":  args.contrast_scale,
        "topk_default":    args.topk_default,
        "topk_candidates": sorted(set(args.topk)),
        # GMM weights (same format as pooled contrastive_preference)
        "clean_means":     model.clean.means.data.cpu(),
        "clean_log_stds":  model.clean.log_stds.data.cpu(),
        "clean_logits":    model.clean.logits.data.cpu(),
        "avoid_means":     model.avoid.means.data.cpu(),
        "avoid_log_stds":  model.avoid.log_stds.data.cpu(),
        "avoid_logits":    model.avoid.logits.data.cpu(),
        "diagnostics": {
            "clean_frames_encoded": clean_tokens.shape[0],
            "avoid_frames_encoded": avoid_tokens.shape[0],
            "clean_token_samples":  clean_flat.shape[0],
            "avoid_token_samples":  avoid_flat.shape[0],
        },
    }
    if pos_stats_saved is not None:
        tcp["position_stats"] = pos_stats_saved

    ckpt["token_contrastive_preference"] = tcp

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)

    print(f"\nSaved -> {args.output}")
    print(f"  Checkpoint keys: {list(ckpt.keys())}")
    print("  'contrastive_preference' (pooled) unchanged if present.")
    print("  'token_contrastive_preference' (token GMM) is the new key.")
    if pos_stats_saved is not None:
        print(
            f"  position_stats saved: "
            f"mu_pos {tuple(pos_stats_saved['mu_pos'].shape)}  "
            f"std_pos {tuple(pos_stats_saved['std_pos'].shape)}"
        )


if __name__ == "__main__":
    main()
