"""Fit Token VampPrior-like preference for TokenViT.

Unlike the token GMM scripts that work on concat(deter,stoch) 320D features,
this script extracts the token posterior distribution (token_mean, token_std,
each (B,64,64)) and fits a per-position K-component Gaussian mixture via
KMeans on posterior means.

Scoring uses the q-integrated density:

    E_{z~q}[log N(z; mu_k, sigma_k^2)]
      = log N(q_mean; mu_k, sqrt(sigma_k^2 + q_std^2))

which accounts for posterior uncertainty when evaluating preference.

Checkpoint key: token_vampprior_preference
Existing keys (contrastive_preference, token_contrastive_preference) are untouched.

Usage:
    ~/.local/bin/uv run python scripts/fit_token_vampprior_preference.py \\
        --checkpoint outputs/token_vit/best.pt \\
        --clean_data data/expert_data_town04.h5 \\
        --obstacle_data data/expert_data_v4.h5 \\
        --output outputs/token_vit/best_vampprior.pt \\
        --config configs/experiment/token_vit.yaml \\
        --K_clean 5 --K_avoid 7 \\
        --max_frames 5000 --min_std 0.01 \\
        --contrast_scale 1.0 \\
        --topk 1 4 8 16 --topk_default 4 \\
        --score_mode q_integrated
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.training.token_vampprior_preference import (
    TokenVampPriorPreference,
    extract_token_posterior_stats,
    fit_position_posterior_mixture,
    log_prob_posterior_under_mixture,
    log_prob_mean_only,
    topk_mean_score,
)


# ---------------------------------------------------------------------------
# Posterior encoding
# ---------------------------------------------------------------------------

def encode_token_posteriors(
    agent: DeepAIFAgent,
    data_path: str,
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode HDF5 frames into per-token posterior statistics.

    Returns:
        all_mu  : (N_frames, N_tokens, Z)  — token_mean on CPU
        all_std : (N_frames, N_tokens, Z)  — token_std  on CPU
    """
    wm  = agent.world_model
    dev = agent._device

    with h5py.File(data_path, "r") as f:
        img_key = "images" if "images" in f else "frames"
        n_avail = min(len(f[img_key]), max_frames)
        images = torch.tensor(f[img_key][:n_avail], dtype=torch.float32)
        states = torch.tensor(f["states"][:n_avail], dtype=torch.float32)

    all_mu:  list[torch.Tensor] = []
    all_std: list[torch.Tensor] = []
    t0 = time.time()

    with torch.no_grad():
        state    = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, agent._cfg.cem.action_dim, device=dev)

        for i in range(len(images)):
            img   = images[i : i + 1].to(dev)
            st    = states[i : i + 1].to(dev)
            embed = wm.encode_obs(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)

            mu, std = extract_token_posterior_stats(post)  # each (1, N, Z)
            all_mu.append(mu.squeeze(0).cpu())             # (N, Z)
            all_std.append(std.squeeze(0).cpu())           # (N, Z)

            state = type(post)(*[x.detach() for x in post])

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(images)} frames  ({elapsed:.1f}s)")

    result_mu  = torch.stack(all_mu)   # (N_frames, N_tokens, Z)
    result_std = torch.stack(all_std)  # (N_frames, N_tokens, Z)
    elapsed = time.time() - t0
    print(
        f"  Encoded {result_mu.shape[0]} frames × "
        f"{result_mu.shape[1]} tokens × {result_mu.shape[2]}d  "
        f"in {elapsed:.1f}s"
    )
    return result_mu, result_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit Token VampPrior-like position-conditional preference",
    )

    # Required
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--clean_data",    required=True)
    parser.add_argument("--obstacle_data", required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--config",        required=True)

    # Mixture
    parser.add_argument("--K_clean",       type=int,   default=5)
    parser.add_argument("--K_avoid",       type=int,   default=7)
    parser.add_argument("--min_std",       type=float, default=0.01)
    parser.add_argument("--seed",          type=int,   default=42)

    # Data
    parser.add_argument("--max_frames",    type=int,   default=5000)

    # Scoring
    parser.add_argument("--contrast_scale", type=float, default=1.0)
    parser.add_argument("--topk",           type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--topk_default",   type=int, default=4)
    parser.add_argument(
        "--score_mode", default="q_integrated",
        choices=["q_integrated", "mean_only"],
        help="q_integrated: integrate over posterior q(z) (default/proposed). "
             "mean_only: evaluate at q_mean point estimate (ablation).",
    )

    # Precision scaling
    parser.add_argument(
        "--q_std_scale", type=float, default=1.0,
        help="Scale applied to q_std before squaring in effective_var. "
             "1.0=standard q_integrated; 0.0=ignore posterior uncertainty.",
    )
    parser.add_argument(
        "--proto_std_scale", type=float, default=1.0,
        help="Scale applied to proto_std before squaring in effective_var. "
             "<1.0 sharpens the prototype mixture; >1.0 smooths it.",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # ----- Load agent -----
    print(f"Loading config: {args.config}")
    cfg   = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    if agent.world_model._wm_type != "token_vit":
        print(
            "ERROR: Token VampPrior-like preference requires TokenViT state "
            "with token_mean/token_std (B,N,Z).\n"
            f"  Got world_model_type={agent.world_model._wm_type!r}.\n"
            "  Use fit_contrastive_preference.py for RSSM checkpoints.",
            file=sys.stderr,
        )
        sys.exit(1)

    tv = cfg.token_vit
    Z  = tv.stoch_dim
    N  = tv.num_tokens

    print(f"  world_model_type : {agent.world_model._wm_type}")
    print(f"  score_mode       : {args.score_mode}")
    print(f"  q_std_scale      : {args.q_std_scale}")
    print(f"  proto_std_scale  : {args.proto_std_scale}")
    print(f"  latent_dim (Z)   : {Z}")
    print(f"  num_tokens (N)   : {N}")
    print(f"  K_clean / K_avoid: {args.K_clean} / {args.K_avoid}")

    # ----- Encode posterior stats -----
    print(f"\nEncoding clean posteriors from {args.clean_data}...")
    clean_mu, clean_std = encode_token_posteriors(
        agent, args.clean_data, args.max_frames,
    )  # each (F_c, N, Z)

    print(f"\nEncoding obstacle posteriors from {args.obstacle_data}...")
    avoid_mu, avoid_std = encode_token_posteriors(
        agent, args.obstacle_data, args.max_frames,
    )  # each (F_a, N, Z)

    # Clamp std to be positive (defensive)
    clean_std = clean_std.clamp(min=args.min_std)
    avoid_std = avoid_std.clamp(min=args.min_std)

    print(f"\nPosterior statistics:")
    print(f"  clean: {clean_mu.shape[0]} frames  "
          f"mu_abs_mean={clean_mu.abs().mean():.4f}  "
          f"std_mean={clean_std.mean():.4f}")
    print(f"  avoid: {avoid_mu.shape[0]} frames  "
          f"mu_abs_mean={avoid_mu.abs().mean():.4f}  "
          f"std_mean={avoid_std.mean():.4f}")

    # ----- Fit per-position mixtures -----
    print(f"\nFitting clean mixture (K={args.K_clean}) over {N} positions...")
    t0 = time.time()
    clean_pm, clean_pls, clean_pl = fit_position_posterior_mixture(
        clean_mu, clean_std, K=args.K_clean, seed=args.seed, min_std=args.min_std,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nFitting obstacle mixture (K={args.K_avoid}) over {N} positions...")
    t0 = time.time()
    avoid_pm, avoid_pls, avoid_pl = fit_position_posterior_mixture(
        avoid_mu, avoid_std, K=args.K_avoid, seed=args.seed, min_std=args.min_std,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ----- Build model -----
    model = TokenVampPriorPreference(
        proto_mean_clean=clean_pm,
        proto_log_std_clean=clean_pls,
        proto_logits_clean=clean_pl,
        proto_mean_avoid=avoid_pm,
        proto_log_std_avoid=avoid_pls,
        proto_logits_avoid=avoid_pl,
        contrast_scale=args.contrast_scale,
        score_mode=args.score_mode,
        topk_default=args.topk_default,
        topk_candidates=sorted(set(args.topk)),
        q_std_scale=args.q_std_scale,
        proto_std_scale=args.proto_std_scale,
    )

    # ----- Diagnostics (per-token) -----
    dev = agent._device
    model.to(dev)

    n_diag = min(200, clean_mu.shape[0], avoid_mu.shape[0])
    c_mu_d = clean_mu[:n_diag].to(dev)
    c_sd_d = clean_std[:n_diag].to(dev)
    a_mu_d = avoid_mu[:n_diag].to(dev)
    a_sd_d = avoid_std[:n_diag].to(dev)

    with torch.no_grad():
        c_tok = model.token_scores(c_mu_d, c_sd_d)  # (n_diag, N)
        a_tok = model.token_scores(a_mu_d, a_sd_d)  # (n_diag, N)

    c_avg = c_tok.mean().item()
    a_avg = a_tok.mean().item()
    gap   = c_avg - a_avg

    print(f"\nToken VampPrior-like diagnostic (per-token, mode={args.score_mode}):")
    print(f"  Clean posterior avg score: {c_avg:+.2f}  (should be > 0)")
    print(f"  Avoid posterior avg score: {a_avg:+.2f}  (should be < 0)")
    print(f"  Gap:                       {gap:+.2f}")

    # Top-k clean-preference diagnostics
    print(f"\nTop-k clean-preference diagnostics:")
    with torch.no_grad():
        for k in sorted(set(args.topk)):
            c_frame = model.score_frames(c_mu_d, c_sd_d, topk=k, direction="clean")
            a_frame = model.score_frames(a_mu_d, a_sd_d, topk=k, direction="clean")
            c_m = c_frame.mean().item()
            a_m = a_frame.mean().item()
            g   = c_m - a_m
            status = "OK" if g > 0 else "WARN: gap <= 0"
            print(f"  k={k:3d}:  clean={c_m:+7.2f}  avoid={a_m:+7.2f}  "
                  f"gap={g:+7.2f}  {status}")

    # Top-k obstacle-energy diagnostics
    print(f"\nTop-k obstacle-energy diagnostics (avoid energy should be higher):")
    with torch.no_grad():
        for k in sorted(set(args.topk)):
            c_frame = model.score_frames(c_mu_d, c_sd_d, topk=k, direction="obstacle")
            a_frame = model.score_frames(a_mu_d, a_sd_d, topk=k, direction="obstacle")
            c_m = c_frame.mean().item()
            a_m = a_frame.mean().item()
            g   = a_m - c_m   # avoid energy should exceed clean energy
            status = "OK" if g > 0 else "WARN: gap <= 0"
            print(f"  k={k:3d}:  clean_energy={c_m:+7.2f}  avoid_energy={a_m:+7.2f}  "
                  f"gap_avoid_minus_clean={g:+7.2f}  {status}")

    # ----- Save -----
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)

    tvp: dict = {
        "type":            "position_conditioned_token_posterior_mixture",
        "method":          "vampprior_like",
        "score_mode":      args.score_mode,
        "latent_dim":      Z,
        "num_tokens":      N,
        "K_clean":         args.K_clean,
        "K_avoid":         args.K_avoid,
        "contrast_scale":  args.contrast_scale,
        "min_std":         args.min_std,
        "q_std_scale":     args.q_std_scale,
        "proto_std_scale": args.proto_std_scale,
        # per-position prototype tensors
        "clean_mean":      clean_pm.cpu(),
        "clean_log_std":   clean_pls.cpu(),
        "clean_logits":    clean_pl.cpu(),
        "avoid_mean":      avoid_pm.cpu(),
        "avoid_log_std":   avoid_pls.cpu(),
        "avoid_logits":    avoid_pl.cpu(),
        "topk_default":    args.topk_default,
        "topk_candidates": sorted(set(args.topk)),
        "diagnostics": {
            "clean_frames_encoded": clean_mu.shape[0],
            "avoid_frames_encoded": avoid_mu.shape[0],
            "clean_avg_score":      c_avg,
            "avoid_avg_score":      a_avg,
            "separation_gap":       gap,
        },
    }

    ckpt["token_vampprior_preference"] = tvp

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)

    print(f"\nSaved -> {args.output}")
    print(f"  Checkpoint keys: {list(ckpt.keys())}")
    print("  'contrastive_preference' and 'token_contrastive_preference' unchanged if present.")
    print("  'token_vampprior_preference' is the new key.")
    print(f"  clean_mean shape: {tvp['clean_mean'].shape}")
    print(f"  avoid_mean shape: {tvp['avoid_mean'].shape}")


if __name__ == "__main__":
    main()
