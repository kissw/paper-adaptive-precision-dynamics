#!/usr/bin/env python3
"""Obstacle linear probe: within-file obstacle_visible separation.

Uses a SINGLE obstacle HDF5 file.  Labels come from the per-frame
obstacle_visible flag — no domain leakage from comparing two different files.

Probe modes (per model):
    pooled       — posterior mean (B, Z), all models
    token_max    — per-token logistic regression → frame score = max over tokens
    token_topk   — per-token LR → frame score = top-k mean (default k=4)
    (token modes are ViT-only; RSSM has no token structure)

Split: episode-level 80/20, repeated n_splits times for variance.

Rollout probe: h prior steps → obstacle separability over imagination horizon.

Usage:
    uv run python scripts/obstacle_probe.py \\
        --data data/expert_obstacle_v5.h5 \\
        --max_per_class 500 \\
        --horizons 0,1,3,5,7,10,15,20 \\
        --token_probe \\
        --topk 4 \\
        --output_dir outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _early_worker_limit(default: int = 10) -> int:
    """Parse --worker from argv and cap BLAS/OpenMP threads BEFORE importing
    numpy/torch.  BLAS thread pools read these env vars at import time, so the
    cap must be applied here — setting them later has no effect.
    """
    w = default
    if "--worker" in sys.argv:
        try:
            w = int(sys.argv[sys.argv.index("--worker") + 1])
        except (ValueError, IndexError):
            pass
    w = max(1, w)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(w)
    return w


_N_WORKERS = _early_worker_limit()

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

# Cap torch intra-op threads to match the requested worker count.
torch.set_num_threads(_N_WORKERS)

sys.path.insert(0, str(Path(__file__).parent))
from pp_gap_v5 import build_world_model


# ─────────────────────────────────────────────────────────────────────────────
# Default model specs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RUNS = [
    {"name": "rssm_off", "config": "configs/default.yaml",
     "checkpoint": "outputs/v6_rssm_off/checkpoints/best.pt", "overrides": []},
    {"name": "rssm_on",  "config": "configs/default.yaml",
     "checkpoint": "outputs/v6_rssm_on/checkpoints/best.pt",  "overrides": []},
    {"name": "vit_off",  "config": "configs/experiment/token_vit.yaml",
     "checkpoint": "outputs/v6_vit_off/checkpoints/best.pt",  "overrides": []},
    {"name": "vit_on",   "config": "configs/experiment/token_vit.yaml",
     "checkpoint": "outputs/v6_vit_on/checkpoints/best.pt",   "overrides": []},
    {"name": "vit_a1",   "config": "configs/experiment/token_vit.yaml",
     "checkpoint": "outputs/v6_vit_a1/checkpoints/best.pt",
     "overrides": ["token_vit.use_action_warp=true"]},
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, max_per_class: int, seed: int = 42):
    """Load balanced frame indices from the obstacle HDF5."""
    with h5py.File(path, "r") as f:
        obs_vis     = f["obstacle_visible"][:].astype(bool)
        ep_ids_all  = f["episode_ids"][:]

    pos_idx = np.where(obs_vis)[0]
    neg_idx = np.where(~obs_vis)[0]
    rng = np.random.default_rng(seed)
    n   = min(len(pos_idx), len(neg_idx), max_per_class)
    pos_sel = rng.choice(pos_idx, n, replace=False)
    neg_sel = rng.choice(neg_idx, n, replace=False)
    sel     = np.sort(np.concatenate([pos_sel, neg_sel]))

    with h5py.File(path, "r") as f:
        images  = torch.tensor(f["images"][sel],  dtype=torch.float32)
        states  = torch.tensor(f["states"][sel],  dtype=torch.float32)
        actions = torch.tensor(f["actions"][sel], dtype=torch.float32)

    ep_ids = ep_ids_all[sel]
    labels = obs_vis[sel].astype(np.int8)
    print(f"  Selected {n} pos + {n} neg from {len(np.unique(ep_ids))} episodes")
    return images, states, actions, ep_ids, labels, sel


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _det_normal():
    orig = Normal.rsample
    Normal.rsample = lambda self, sample_shape=(): self.loc
    try:
        yield
    finally:
        Normal.rsample = orig


def _make_det(state):
    if hasattr(state, "token_mean"):
        return type(state)(deter=state.deter, stoch=state.token_mean,
                           mean=state.mean, std=state.std,
                           token_mean=state.token_mean, token_std=state.token_std)
    return type(state)(deter=state.deter, stoch=state.mean, mean=state.mean, std=state.std)


def _action_dim(wm) -> int:
    r = wm.rssm
    if hasattr(r, "_action_dim"):
        return r._action_dim
    return r._img_in[0].in_features - r._stoch_dim


def _is_vit(state) -> bool:
    return hasattr(state, "token_mean") and state.deter.ndim == 3


def _token_feat(state) -> np.ndarray | None:
    """Extract (N_tok, D+Z) per-token features from a single state."""
    if not _is_vit(state):
        return None
    tok = torch.cat([state.deter, state.stoch], dim=-1)  # (1, N_tok, D+Z)
    return tok.squeeze(0).cpu().numpy()                   # (N_tok, D+Z)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding with context
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_with_context(
    wm,
    path: str,
    selected_global_idx: np.ndarray,
    episode_ids_all: np.ndarray,
    obs_vis_all: np.ndarray,
    context_len: int,
    device: torch.device,
) -> tuple[np.ndarray, list, np.ndarray | None]:
    """Encode each frame with short context; return pooled feats, states, token feats.

    Returns:
        pooled_feats  : (N, Z)
        post_list     : N RSSMState objects
        token_feats   : (N, N_tok, D+Z) for ViT, None for RSSM
    """
    action_dim = _action_dim(wm)
    pooled     = []
    tokens     = []
    post_list  = []

    with h5py.File(path, "r") as f:
        for idx in selected_global_idx:
            ep = int(episode_ids_all[idx])
            ep_start = int(np.where(episode_ids_all == ep)[0][0])
            ctx_start = max(ep_start, idx - context_len + 1)

            s = slice(ctx_start, idx + 1)
            ctx_img = torch.tensor(f["images"][s],  dtype=torch.float32)
            ctx_st  = torch.tensor(f["states"][s],  dtype=torch.float32)
            ctx_act = torch.tensor(f["actions"][s], dtype=torch.float32)

            state    = wm.rssm.initial(1, device)
            prev_act = torch.zeros(1, action_dim, device=device)
            with _det_normal():
                for t in range(len(ctx_img)):
                    img   = ctx_img[t:t+1].to(device)
                    st    = ctx_st[t:t+1].to(device)
                    act   = ctx_act[t:t+1].to(device)
                    embed = wm.encode_obs(img, st)
                    post, _ = wm.rssm.obs_step(state, prev_act, embed)
                    state   = _make_det(post)
                    prev_act = act

            pooled.append(state.mean.squeeze(0).cpu().numpy())
            tf = _token_feat(state)
            if tf is not None:
                tokens.append(tf)
            post_list.append(type(state)(*[x.detach() for x in state]))

    token_feats = np.stack(tokens) if tokens else None  # (N, N_tok, D+Z)
    return np.stack(pooled), post_list, token_feats


# ─────────────────────────────────────────────────────────────────────────────
# Rollout: pooled + token in one pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_features_both(
    wm,
    initial_states: list,
    h: int,
    device: torch.device,
    return_tokens: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Roll out h zero-action prior steps.

    Returns (pooled_feats (N, Z), token_feats (N, N_tok, D+Z) or None).
    Computing both in one pass avoids double rollout cost.
    """
    action_dim = _action_dim(wm)
    pooled = []
    tokens = []
    with _det_normal():
        for s in initial_states:
            state = s
            for _ in range(h):
                act   = torch.zeros(1, action_dim, device=device)
                state = wm.rssm.img_step(state, act)
                state = _make_det(state)
            pooled.append(state.mean.squeeze(0).cpu().numpy())
            if return_tokens:
                tf = _token_feat(state)
                if tf is not None:
                    tokens.append(tf)
    token_arr = np.stack(tokens) if tokens else None
    return np.stack(pooled), token_arr


# ─────────────────────────────────────────────────────────────────────────────
# Probe: pooled (existing) + token-level (new)
# ─────────────────────────────────────────────────────────────────────────────

def _balance(feats, labels, max_n, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n   = min(len(pos), len(neg), max_n)
    if n == 0:
        return feats, labels
    sel = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, n, replace=False)])
    return feats[sel], labels[sel]


def _balance_with_idx(full_mask, labels, max_n, seed):
    """Returns frame indices (into the full N-element array) + selected labels."""
    rng     = np.random.default_rng(seed)
    idx_all = np.where(full_mask)[0]
    pos     = idx_all[labels[full_mask] == 1]
    neg     = idx_all[labels[full_mask] == 0]
    n       = min(len(pos), len(neg), max_n)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=np.int8)
    pos_s   = rng.choice(pos, n, replace=False)
    neg_s   = rng.choice(neg, n, replace=False)
    sel_idx = np.concatenate([pos_s, neg_s])
    sel_lbl = np.array([1] * n + [0] * n, dtype=np.int8)
    return sel_idx, sel_lbl


def probe_episode_split(feats, labels, ep_ids, n_splits=5, max_per_class=400):
    """Episode-level split logistic regression probe on pooled features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    unique_eps = np.unique(ep_ids)
    accs, aucs = [], []
    for seed in range(n_splits):
        rng      = np.random.default_rng(seed * 137 + 42)
        perm     = rng.permutation(unique_eps)
        test_set = set(perm[:max(1, int(len(perm) * 0.2))].tolist())
        tr_mask  = np.array([e not in test_set for e in ep_ids])
        te_mask  = ~tr_mask

        X_tr, y_tr = _balance(feats[tr_mask], labels[tr_mask], max_per_class, seed)
        X_te, y_te = _balance(feats[te_mask], labels[te_mask], max_per_class, seed)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf  = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
        aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

    if not accs:
        return {"acc": float("nan"), "acc_std": 0.0,
                "auc": float("nan"), "auc_std": 0.0}
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def probe_token_level(
    token_feats: np.ndarray,
    labels: np.ndarray,
    ep_ids: np.ndarray,
    n_splits: int = 5,
    max_per_class: int = 400,
    topk: int = 4,
) -> dict[str, dict[str, float]]:
    """Token-level probe with max and top-k mean aggregation.

    token_feats : (N, N_tok, D)
    labels      : (N,) 0/1
    ep_ids      : (N,)

    Train: flatten (N_tr * N_tok, D), label = repeat(frame_label, N_tok).
           → probe learns which token patterns predict visibility.
    Test : predict per-token probabilities, reshape to (N_te, N_tok),
           aggregate per frame, compare to frame labels.

    Episode-level split strictly observed: tokens from the same frame
    belong to the same split as the frame itself.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    N, N_tok, D = token_feats.shape
    unique_eps   = np.unique(ep_ids)
    res_max  = {"acc": [], "auc": []}
    res_topk = {"acc": [], "auc": []}
    topk_actual = min(topk, N_tok)

    for seed in range(n_splits):
        rng      = np.random.default_rng(seed * 137 + 42)
        perm     = rng.permutation(unique_eps)
        test_set = set(perm[:max(1, int(len(perm) * 0.2))].tolist())
        tr_mask  = np.array([e not in test_set for e in ep_ids])
        te_mask  = ~tr_mask

        tr_idx, tr_lbl = _balance_with_idx(tr_mask, labels, max_per_class, seed)
        te_idx, te_lbl = _balance_with_idx(te_mask, labels, max_per_class, seed)
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        if len(np.unique(tr_lbl)) < 2 or len(np.unique(te_lbl)) < 2:
            continue

        # Expand tokens for training: (N_tr * N_tok, D)
        X_tr = token_feats[tr_idx].reshape(-1, D)      # (N_tr * N_tok, D)
        y_tr = np.repeat(tr_lbl, N_tok)                 # (N_tr * N_tok,)
        X_te = token_feats[te_idx].reshape(-1, D)       # (N_te * N_tok, D)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(X_tr, y_tr)

        p_tok  = clf.predict_proba(X_te)[:, 1]          # (N_te * N_tok,)
        p_frame = p_tok.reshape(-1, N_tok)               # (N_te, N_tok)

        # max aggregation
        p_max = p_frame.max(axis=1)
        res_max["acc"].append(accuracy_score(te_lbl, (p_max > 0.5).astype(int)))
        res_max["auc"].append(roc_auc_score(te_lbl, p_max))

        # top-k mean aggregation
        p_topk = np.sort(p_frame, axis=1)[:, -topk_actual:].mean(axis=1)
        res_topk["acc"].append(accuracy_score(te_lbl, (p_topk > 0.5).astype(int)))
        res_topk["auc"].append(roc_auc_score(te_lbl, p_topk))

    def _agg(d):
        if not d["acc"]:
            return {"acc": float("nan"), "acc_std": 0.0,
                    "auc": float("nan"), "auc_std": 0.0}
        return {"acc": float(np.mean(d["acc"])), "acc_std": float(np.std(d["acc"])),
                "auc": float(np.mean(d["auc"])), "auc_std": float(np.std(d["auc"]))}

    return {"token_max": _agg(res_max), f"token_top{topk}": _agg(res_topk)}


# ─────────────────────────────────────────────────────────────────────────────
# Plot: multi-mode
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves_multi(
    results: dict,        # {model_name: {mode: {h: metrics}}}
    horizons: list[int],
    output_png: str,
    topk: int = 4,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Colour per model, linestyle per mode
    COLORS = {"rssm_off": "#5599dd", "rssm_on": "#1144aa",
               "vit_off": "#ee8833", "vit_on": "#cc5500", "vit_a1": "#882200"}
    LS     = {"pooled": "--", "token_max": "-", f"token_top{topk}": ":"}
    MARKER = {"pooled": "o",  "token_max": "s", f"token_top{topk}": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    legend_handles = []

    for metric, ax, ylabel in [("acc", axes[0], "Accuracy"),
                                ("auc", axes[1], "AUC-ROC")]:
        for model_name, mode_data in results.items():
            col = COLORS.get(model_name, "#888")
            for mode, hdata in mode_data.items():
                hs  = sorted(hdata.keys())
                ys  = [hdata[h][metric]          for h in hs]
                err = [hdata[h][f"{metric}_std"] for h in hs]
                ls  = LS.get(mode, "-")
                mk  = MARKER.get(mode, "o")
                lbl = f"{model_name}/{mode}"
                h_line, = ax.plot(hs, ys, ls=ls, marker=mk, color=col,
                                  label=lbl, linewidth=1.5, markersize=4,
                                  alpha=0.85)
                ax.fill_between(hs, [y-e for y,e in zip(ys,err)],
                                     [y+e for y,e in zip(ys,err)],
                                alpha=0.10, color=col)
                if metric == "acc" and lbl not in [h.get_label() for h in legend_handles]:
                    legend_handles.append(h_line)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("Rollout horizon h", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Obstacle-visible probe — {ylabel}", fontsize=11)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.02)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Obstacle-visible probe: pooled + token-level modes",
    )
    parser.add_argument("--data", default="data/expert_obstacle_v5.h5")
    parser.add_argument("--horizons",      default="0,1,3,5,7,10,15,20")
    parser.add_argument("--max_per_class", type=int, default=500)
    parser.add_argument("--context_len",   type=int, default=5)
    parser.add_argument("--n_splits",      type=int, default=5)
    parser.add_argument("--topk",          type=int, default=4,
                        help="k for top-k token aggregation (default 4)")
    parser.add_argument("--token_probe",   action="store_true", default=False,
                        help="Enable token-level probe for ViT models")
    parser.add_argument("--runs_json", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--worker", type=int, default=10,
        help="Max CPU threads for BLAS/torch/sklearn (default 10). "
             "Caps OMP/MKL/OpenBLAS thread pools to limit CPU usage.",
    )
    args = parser.parse_args()

    # Worker count was already applied to BLAS env vars at import time via
    # _early_worker_limit().  Re-assert here for torch + sklearn n_jobs.
    n_workers = max(1, args.worker)
    torch.set_num_threads(n_workers)
    print(f"CPU thread cap: {n_workers} workers "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")

    horizons = [int(h) for h in args.horizons.split(",")]
    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = DEFAULT_RUNS if not args.runs_json else json.load(open(args.runs_json))
    runs = [r for r in runs if Path(r["checkpoint"]).exists()]
    if not runs:
        print("ERROR: no valid checkpoints"); sys.exit(1)

    print(f"Loading metadata from {args.data} …")
    with h5py.File(args.data, "r") as f:
        episode_ids_all = f["episode_ids"][:]
        obs_vis_all     = f["obstacle_visible"][:].astype(bool)
    print(f"  pos={obs_vis_all.sum()}  neg={(~obs_vis_all).sum()}"
          f"  episodes={len(np.unique(episode_ids_all))}")

    images, states, actions, ep_ids, labels, sel_idx = load_data(
        args.data, args.max_per_class,
    )

    # results[model_name][mode][h] = {acc, acc_std, auc, auc_std}
    results: dict[str, dict[str, dict[int, dict]]] = {}

    for spec in runs:
        name = spec["name"]
        print(f"\n[{name}]")
        wm, _, cfg = build_world_model(
            spec["config"], spec["checkpoint"], device,
            overrides=spec.get("overrides") or None,
        )
        wm._cfg = cfg
        is_vit_model = getattr(
            getattr(cfg, "model", None), "world_model_type", "rssm"
        ) == "token_vit"
        do_token = args.token_probe and is_vit_model

        print(f"  encoding {len(sel_idx)} frames (ctx={args.context_len},"
              f" token={'yes' if do_token else 'no'}) …", end="", flush=True)
        feats, post_list, tok_feats = encode_with_context(
            wm, args.data, sel_idx, episode_ids_all, obs_vis_all,
            args.context_len, device,
        )
        print(f" done  feat_dim={feats.shape[1]}"
              + (f"  tok_shape={tok_feats.shape}" if tok_feats is not None else ""))

        results[name] = {"pooled": {}}
        if do_token:
            results[name]["token_max"] = {}
            results[name][f"token_top{args.topk}"] = {}

        for h in horizons:
            if h == 0:
                p_feats  = feats
                t_feats  = tok_feats
            else:
                print(f"  rollout h={h:2d} …", end="", flush=True)
                p_feats, t_feats = rollout_features_both(
                    wm, post_list, h, device,
                    return_tokens=do_token,
                )
                print()

            # ── Pooled probe ───────────────────────────────────────────────
            m = probe_episode_split(p_feats, labels, ep_ids,
                                    args.n_splits, args.max_per_class)
            results[name]["pooled"][h] = m
            line = (f"    h={h:2d}  pooled: acc={m['acc']:.3f}±{m['acc_std']:.3f}"
                    f"  AUC={m['auc']:.3f}±{m['auc_std']:.3f}")

            # ── Token probe (ViT only) ─────────────────────────────────────
            if do_token and t_feats is not None:
                tok_res = probe_token_level(
                    t_feats, labels, ep_ids,
                    n_splits=args.n_splits,
                    max_per_class=args.max_per_class,
                    topk=args.topk,
                )
                for mode_key, mode_metrics in tok_res.items():
                    results[name][mode_key][h] = mode_metrics
                best_tok = max(tok_res.values(), key=lambda x: x["auc"])
                line += (f"  | best-tok: acc={best_tok['acc']:.3f}"
                         f"  AUC={best_tok['auc']:.3f}")

            print(line)

    # ── Summary table ─────────────────────────────────────────────────────
    col_w = 14
    for metric_label, key in [("Accuracy", "acc"), ("AUC-ROC", "auc")]:
        print(f"\n{'='*60}")
        print(f"  {metric_label}  (mode=pooled)")
        print(f"{'='*60}")
        print("  " + f"{'h':>2}" + "".join(f"{n:>{col_w}}" for n in results))
        for h in horizons:
            row = f"  {h:>2}"
            for name in results:
                v = results[name]["pooled"][h]
                row += f"  {v[key]:.3f}±{v[key+'_std']:.2f}".rjust(col_w)
            print(row)

    # Token mode summary (ViT only)
    for mode in (f"token_max", f"token_top{args.topk}"):
        vit_with_mode = [n for n in results if mode in results[n]]
        if not vit_with_mode:
            continue
        print(f"\n  {metric_label}  (mode={mode})")
        print("  " + f"{'h':>2}" + "".join(f"{n:>{col_w}}" for n in vit_with_mode))
        for h in horizons:
            row = f"  {h:>2}"
            for name in vit_with_mode:
                v = results[name][mode][h]
                row += f"  {v[key]:.3f}±{v[key+'_std']:.2f}".rjust(col_w)
            print(row)

    # ── Save ──────────────────────────────────────────────────────────────
    json_path = out_dir / "obstacle_probe_results.json"
    with open(json_path, "w") as f:
        json.dump({
            name: {
                mode: {str(h): v for h, v in hd.items()}
                for mode, hd in mode_data.items()
            }
            for name, mode_data in results.items()
        }, f, indent=2)
    print(f"\n  JSON → {json_path}")

    plot_curves_multi(results, horizons,
                      str(out_dir / "obstacle_probe_curve.png"), args.topk)
    print("\nDone.")


if __name__ == "__main__":
    main()
