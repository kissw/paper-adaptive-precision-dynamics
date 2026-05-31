#!/usr/bin/env python3
"""Decoder-free rollout-retention diagnostic for the warp ablation (offline).

WHY THIS, AND WHY NO obstacle_head / beta_obstacle_aux
------------------------------------------------------
At inference our system never decodes images; EFE is computed entirely in
latent space. So the question "does an action-conditioned prior rollout keep
obstacle information alive?" must also be answered in latent space, WITHOUT a
decoder (whose reconstruction quality would otherwise contaminate the verdict).

We also deliberately do NOT use the auxiliary obstacle_head: training with
beta_obstacle_aux > 0 spoon-feeds obstacle presence into the latent, which is
cheating for an evaluation. All three models here are assumed trained with
beta_obstacle_aux = 0. This script touches no model parameters and never trains
anything inside the world model.

We measure two complementary, cheating-free signals.

(D1) Rollout latent tracking error  [cheating 0]
     Ground-truth latent = the filtered POSTERIOR feature obtained by running
     obs_step over the *actual* future frames (this GT encoding is for scoring
     only; it is NEVER fed into the rollout). Rollout latent = prior img_step
     from the same context, actions only. At each horizon h we measure the
     distance between rollout feature and GT-posterior feature, both as L2 and
     cosine distance. We split cases into obstacle (context clear, an obstacle
     appears during rollout: a 0->1 task_label transition) vs clear (all clear),
     and report each group plus the gap = obstacle_err - clear_err.
     Hypothesis: in obstacle cases RSSM's tracking error blows up fast (it
     cannot follow the scene change in imagination) while warp stays lower; the
     decisive quantity is how the obstacle-minus-clear gap differs by model.

(D2) Frozen post-hoc linear probe separation  [sharp signal]
     With the model fully frozen, fit a logistic-regression probe that reads
     task_label (1 = obstacle present) from CONTEXT POSTERIOR features. This is
     standard post-hoc representation probing: it only READS the latent, never
     modifies it, so it is not cheating and is unlike the aux head. The probe is
     fit per model (each latent space differs), on posterior features (obs seen),
     with a held-out split to confirm it actually reads obstacle (val
     separation > 0). The SAME frozen probe is then applied to PRIOR rollout
     features to measure, per horizon h, the obstacle separation
     (mean logit at obstacle-present steps - mean logit at free steps), in nats.
     Hypothesis: separation is high at the posterior, collapses toward 0 with
     horizon for RSSM rollouts, and is retained for warp. This reproduces and
     extends the README's +276 nats finding in latent space, decoder-free.
     If a model's posterior val separation is ~0 the latent carries no obstacle
     signal: that model is flagged "probe weak" and its D2 is untrustworthy.

EXAMPLE (real 3-model run; checkpoints live on the server, do not run here):

  uv run python scripts/diagnose_rollout_retention.py \
    --model rssm:configs/experiment/task_b_v5.yaml:runs/rssm_.../best.pt \
    --model token_vit:configs/experiment/token_vit.yaml:runs/token_vit_.../best.pt \
    --model warp_token_vit:configs/experiment/token_vit_warp.yaml:runs/warp_.../best.pt \
    --data data/expert_data_v4.h5 --num_cases 40 --horizon 15
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Repo + src on path (mirrors compare_rssm_token_vit_rollout.py layout).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_inference.config import Config  # noqa: E402
from active_inference.agent import WorldModel  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn absent
    _HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# Model building / loading (reuses compare_rssm_token_vit_rollout.py structure)
# ---------------------------------------------------------------------------
def parse_model_spec(spec: str):
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"--model must be 'name:config_path:checkpoint_path', got {spec!r}"
        )
    return parts[0], parts[1], ":".join(parts[2:])


def build_world_model(cfg_path: str, ckpt_path: str, device: torch.device):
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("world_model",
                          ckpt.get("model_state",
                                   ckpt.get("state_dict", ckpt)))
        else:
            sd = ckpt
        try:
            wm.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            raise RuntimeError(
                f"parameter shape mismatch while loading {ckpt_path}; the config "
                f"feat dim likely differs from the trained model.\n{e}"
            )
    else:
        print(f"  WARNING: checkpoint not found ({ckpt_path!r}); "
              f"using randomly-initialised weights.")
    wm.eval()
    return wm


# ---------------------------------------------------------------------------
# Latent extraction. obstacle_head is intentionally never referenced.
# get_feat returns (B, 320) for both RSSM and TokenViT.
# ---------------------------------------------------------------------------
def _detach_state(state):
    return type(state)(*[x.detach() for x in state])


@torch.no_grad()
def _feat(wm, state):
    return wm.rssm.get_feat(state).squeeze(0).cpu().numpy()  # (320,)


@torch.no_grad()
def context_state(wm, images, states, actions, start, context_len, device):
    """Run posterior obs_step over the context window; return (state, prev_act).

    prev_act after the loop transitions the last context frame -> first future
    frame (matches compare_rssm_token_vit_rollout.py's action convention).
    """
    T = images.shape[0]
    state = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, actions.shape[-1], device=device)
    for i in range(context_len):
        t = start + i
        embed = wm.encode_obs(images[t:t + 1].to(device), states[t:t + 1].to(device))
        post, _ = wm.rssm.obs_step(state, prev_act, embed)
        state = _detach_state(post)
        prev_act = actions[t:t + 1].to(device) if t < T - 1 else torch.zeros_like(prev_act)
    return state, prev_act


@torch.no_grad()
def posterior_feat_at(wm, images, states, actions, t, context_len, device):
    """Filtered posterior feature for frame t (context_len frames ending at t)."""
    start = t - context_len + 1
    state, _ = context_state(wm, images, states, actions, start, context_len, device)
    return _feat(wm, state)


@torch.no_grad()
def rollout_and_gt(wm, images, states, actions, start, context_len, horizon, device):
    """From one context window return per-horizon GT-posterior and prior features.

    gt_feats[h]   : posterior feature after obs_step on the actual future frame h
                    (scoring target; NEVER fed into the prior rollout).
    prior_feats[h]: feature after prior img_step h (actions only, no observation).
    """
    T = images.shape[0]
    ctx_state, ctx_prev_act = context_state(
        wm, images, states, actions, start, context_len, device)

    # GT posterior future: continue obs_step with the real frames.
    gt_state, gt_prev_act = ctx_state, ctx_prev_act
    gt_feats = []
    for h in range(horizon):
        t = start + context_len + h
        embed = wm.encode_obs(images[t:t + 1].to(device), states[t:t + 1].to(device))
        post, _ = wm.rssm.obs_step(gt_state, gt_prev_act, embed)
        gt_state = _detach_state(post)
        gt_feats.append(_feat(wm, gt_state))
        gt_prev_act = actions[t:t + 1].to(device) if t < T - 1 else torch.zeros_like(gt_prev_act)

    # Prior rollout: img_step only.
    pr_state = ctx_state
    prior_feats = []
    for h in range(horizon):
        t = start + context_len + h - 1   # action transitioning into future step h
        act = actions[t:t + 1].to(device) if t < T - 1 else torch.zeros(1, actions.shape[-1], device=device)
        pr_state = wm.rssm.img_step(pr_state, act)
        prior_feats.append(_feat(wm, pr_state))

    return np.stack(gt_feats), np.stack(prior_feats)  # each (horizon, 320)


# ---------------------------------------------------------------------------
# Frozen post-hoc linear probe (sklearn preferred, torch fallback)
# ---------------------------------------------------------------------------
class LinearProbe:
    """Standardise features then fit logistic regression. Read-only at eval."""

    def __init__(self):
        self.mu = None
        self.sd = None
        self.backend = None
        self.clf = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Xs = (X - self.mu) / self.sd
        if _HAVE_SKLEARN:
            self.clf = LogisticRegression(max_iter=2000, C=1.0)
            self.clf.fit(Xs, y)
            self.backend = "sklearn"
        else:
            self._fit_torch(Xs, y)
            self.backend = "torch"

    def _fit_torch(self, Xs, y):
        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        w = torch.zeros(Xt.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(400):
            opt.zero_grad()
            logit = Xt @ w + b
            loss = lossf(logit, yt) + 1e-3 * (w * w).sum()
            loss.backward()
            opt.step()
        self.w = w.detach().numpy().astype(np.float64)
        self.b = float(b.detach().numpy()[0])

    def logit(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mu) / self.sd
        if self.backend == "sklearn":
            return self.clf.decision_function(Xs)
        return Xs @ self.w + self.b


# ---------------------------------------------------------------------------
# Data + case/window collection
# ---------------------------------------------------------------------------
def load_data(data_path, label_key):
    with h5py.File(data_path, "r") as f:
        for required in ("images", "states", "actions"):
            if required not in f:
                raise KeyError(f"dataset {data_path} missing required key '{required}'")
        if label_key not in f:
            keys = list(f.keys())
            raise KeyError(
                f"dataset {data_path} has no label key '{label_key}'. "
                f"Available keys: {keys}. Pass the correct --label_key."
            )
        images = torch.tensor(np.array(f["images"]), dtype=torch.float32)
        states = torch.tensor(np.array(f["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(f["actions"]), dtype=torch.float32)
        labels = np.asarray(f[label_key]).reshape(-1).astype(int)
        episodes = (np.asarray(f["episode_ids"]).reshape(-1).astype(int)
                    if "episode_ids" in f else np.zeros(len(labels), dtype=int))
    return images, states, actions, labels, episodes


def _same_episode(episodes, a, b):
    seg = episodes[a:b]
    return seg.size > 0 and seg.min() == seg.max()


def collect_cases(labels, episodes, N, context_len, horizon, num_cases, stride):
    """Split windows into obstacle (clear context, 0->1 in rollout) and clear."""
    L = context_len + horizon
    target = max(1, num_cases // 2)
    obstacle, clear = [], []
    for s in range(0, N - L + 1, stride):
        if not _same_episode(episodes, s, s + L):
            continue
        ctx = labels[s:s + context_len]
        roll = labels[s + context_len:s + L]
        if ctx.sum() != 0:
            continue  # require clear context for both groups
        if roll.sum() == 0:
            clear.append(s)
        elif roll[0] == 0 and roll.max() == 1:
            obstacle.append(s)  # obstacle appears during rollout (0->1 transition)
    obstacle = obstacle[:target]
    clear = clear[:target]
    return obstacle, clear


def collect_probe_samples(labels, episodes, N, context_len, exclude_starts,
                          horizon, max_samples=3000, stride=7):
    """Posterior-frame samples (feature target frame t, label[t]) for probe fit.

    Excludes any frame falling inside a held-out case window so probe fit and
    rollout evaluation do not overlap.
    """
    L = context_len + horizon
    excluded = np.zeros(N, dtype=bool)
    for s in exclude_starts:
        excluded[s:s + L] = True
    pos, neg = [], []
    for t in range(context_len - 1, N, stride):
        if excluded[t]:
            continue
        if not _same_episode(episodes, t - context_len + 1, t + 1):
            continue
        (pos if labels[t] == 1 else neg).append(t)
    # balance roughly so the probe sees both classes
    k = min(len(pos), len(neg), max_samples // 2)
    rng = np.random.default_rng(0)
    if k == 0:
        return (pos + neg)[:max_samples]
    pos = list(rng.choice(pos, size=k, replace=False))
    neg = list(rng.choice(neg, size=k, replace=False))
    return pos + neg


# ---------------------------------------------------------------------------
# Per-model diagnostic
# ---------------------------------------------------------------------------
def diagnose_model(wm, images, states, actions, labels, episodes,
                   obstacle_cases, clear_cases, probe_frames,
                   context_len, horizon, device):
    H = horizon

    # ---- D1: tracking error, per case group ----
    def group_errors(starts):
        l2 = np.full((len(starts), H), np.nan)
        cos = np.full((len(starts), H), np.nan)
        for i, s in enumerate(starts):
            gt, pr = rollout_and_gt(wm, images, states, actions, s,
                                    context_len, H, device)
            d = gt - pr
            l2[i] = np.linalg.norm(d, axis=1)
            denom = (np.linalg.norm(gt, axis=1) * np.linalg.norm(pr, axis=1)) + 1e-8
            cos[i] = 1.0 - (gt * pr).sum(axis=1) / denom
        return l2, cos

    obs_l2, obs_cos = group_errors(obstacle_cases)
    clr_l2, clr_cos = group_errors(clear_cases)

    d1 = {
        "obstacle_l2": np.nanmean(obs_l2, axis=0) if len(obstacle_cases) else np.full(H, np.nan),
        "clear_l2": np.nanmean(clr_l2, axis=0) if len(clear_cases) else np.full(H, np.nan),
        "obstacle_cos": np.nanmean(obs_cos, axis=0) if len(obstacle_cases) else np.full(H, np.nan),
        "clear_cos": np.nanmean(clr_cos, axis=0) if len(clear_cases) else np.full(H, np.nan),
        "n_obstacle": len(obstacle_cases),
        "n_clear": len(clear_cases),
    }
    d1["gap_l2"] = d1["obstacle_l2"] - d1["clear_l2"]

    # ---- D2: frozen post-hoc probe ----
    feats = np.stack([posterior_feat_at(wm, images, states, actions, t,
                                        context_len, device) for t in probe_frames]) \
        if probe_frames else np.zeros((0, wm.rssm.get_feat(wm.rssm.initial(1, device)).shape[-1]))
    plabels = np.array([labels[t] for t in probe_frames], dtype=int)

    probe = None
    val_sep = float("nan")
    probe_weak = True
    if len(np.unique(plabels)) == 2 and len(plabels) >= 8:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(plabels))
        n_val = max(2, int(0.2 * len(perm)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        # ensure train has both classes; else fall back to all-train
        if len(np.unique(plabels[tr_idx])) < 2:
            tr_idx = np.arange(len(plabels))
        probe = LinearProbe()
        probe.fit(feats[tr_idx], plabels[tr_idx])
        vlog = probe.logit(feats[val_idx])
        vy = plabels[val_idx]
        if (vy == 1).any() and (vy == 0).any():
            val_sep = float(vlog[vy == 1].mean() - vlog[vy == 0].mean())
        probe_weak = not (np.isfinite(val_sep) and val_sep >= 0.5)

    # D2 separation across horizon: apply frozen probe to prior rollout feats,
    # pooled over all cases, grouped by the GT future label at each step.
    d2_sep = np.full(H, np.nan)
    d2_n = np.zeros(H, dtype=int)
    if probe is not None:
        present = [[] for _ in range(H)]
        free = [[] for _ in range(H)]
        for s in (obstacle_cases + clear_cases):
            _, pr = rollout_and_gt(wm, images, states, actions, s,
                                   context_len, H, device)
            logit = probe.logit(pr)  # (H,)
            for h in range(H):
                lab = labels[s + context_len + h]
                (present[h] if lab == 1 else free[h]).append(logit[h])
        for h in range(H):
            if present[h] and free[h]:
                d2_sep[h] = float(np.mean(present[h]) - np.mean(free[h]))
                d2_n[h] = len(present[h]) + len(free[h])

    return {
        "d1": d1,
        "d2_sep": d2_sep,
        "d2_n": d2_n,
        "val_sep": val_sep,
        "probe_weak": probe_weak,
        "probe_backend": (probe.backend if probe is not None else "none"),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(out_csv, results, horizon):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "horizon_step", "group", "value", "n"])
        for name, r in results.items():
            d1 = r["d1"]
            for h in range(horizon):
                w.writerow([name, "d1_l2", h + 1, "obstacle",
                            _fmt(d1["obstacle_l2"][h]), d1["n_obstacle"]])
                w.writerow([name, "d1_l2", h + 1, "clear",
                            _fmt(d1["clear_l2"][h]), d1["n_clear"]])
                w.writerow([name, "d1_cos", h + 1, "obstacle",
                            _fmt(d1["obstacle_cos"][h]), d1["n_obstacle"]])
                w.writerow([name, "d1_cos", h + 1, "clear",
                            _fmt(d1["clear_cos"][h]), d1["n_clear"]])
                w.writerow([name, "d1_gap", h + 1, "all",
                            _fmt(d1["gap_l2"][h]), d1["n_obstacle"]])
            # posterior val separation sanity row (horizon_step 0)
            w.writerow([name, "d2_separation", 0, "posterior_val",
                        _fmt(r["val_sep"]),
                        "weak" if r["probe_weak"] else "ok"])
            for h in range(horizon):
                w.writerow([name, "d2_separation", h + 1, "all",
                            _fmt(r["d2_sep"][h]), int(r["d2_n"][h])])
    print(f"\nWrote CSV: {out_csv}")


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.6f}"


def print_tables(results, horizon):
    names = list(results.keys())
    print("\n=== D1 gap (obstacle L2 err - clear L2 err) vs horizon ===")
    print("h   " + "  ".join(f"{n:>16}" for n in names))
    for h in range(horizon):
        cells = [f"{results[n]['d1']['gap_l2'][h]:>16.3f}"
                 if np.isfinite(results[n]['d1']['gap_l2'][h]) else f"{'n/a':>16}"
                 for n in names]
        print(f"{h + 1:<3} " + "  ".join(cells))

    print("\n=== D2 probe separation (nats) vs horizon ===")
    print("post: " + "  ".join(
        f"{n}={results[n]['val_sep']:.2f}{'(weak)' if results[n]['probe_weak'] else ''}"
        for n in names))
    print("h   " + "  ".join(f"{n:>16}" for n in names))
    for h in range(horizon):
        cells = [f"{results[n]['d2_sep'][h]:>16.3f}"
                 if np.isfinite(results[n]['d2_sep'][h]) else f"{'n/a':>16}"
                 for n in names]
        print(f"{h + 1:<3} " + "  ".join(cells))


def plot_all(results, horizon, out_dir):
    xs = list(range(1, horizon + 1))

    # (1) D1 tracking error vs horizon: obstacle solid, clear dashed.
    plt.figure(figsize=(7, 5))
    for n, r in results.items():
        line, = plt.plot(xs, r["d1"]["obstacle_l2"], marker="o", label=f"{n} obstacle")
        plt.plot(xs, r["d1"]["clear_l2"], marker="x", ls="--",
                 color=line.get_color(), label=f"{n} clear")
    plt.xlabel("rollout horizon step")
    plt.ylabel("rollout-vs-GT-posterior L2 distance")
    plt.title("D1: latent tracking error vs horizon")
    plt.legend(fontsize=7)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "d1_tracking_error.png")
    plt.savefig(p1, dpi=120)
    plt.close()

    # (2) D1 gap vs horizon.
    plt.figure(figsize=(7, 5))
    for n, r in results.items():
        plt.plot(xs, r["d1"]["gap_l2"], marker="o", label=n)
    plt.axhline(0.0, color="gray", lw=0.8, ls="--")
    plt.xlabel("rollout horizon step")
    plt.ylabel("obstacle L2 err - clear L2 err")
    plt.title("D1 gap vs horizon\n(larger = obstacle scene harder to imagine)")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "d1_gap.png")
    plt.savefig(p2, dpi=120)
    plt.close()

    # (3) D2 separation vs horizon.
    plt.figure(figsize=(7, 5))
    for n, r in results.items():
        lbl = n + (" (probe weak)" if r["probe_weak"] else "")
        plt.plot(xs, r["d2_sep"], marker="o", label=lbl)
    plt.axhline(0.0, color="gray", lw=0.8, ls="--")
    plt.xlabel("rollout horizon step")
    plt.ylabel("probe obstacle separation (present - free) [nats]")
    plt.title("D2: frozen-probe separation vs horizon\n"
              "(warp hypothesis: retained for warp, collapses for RSSM)")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "d2_probe_separation.png")
    plt.savefig(p3, dpi=120)
    plt.close()
    print(f"Wrote figures: {p1}, {p2}, {p3}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="Repeatable. Format 'name:config_path:checkpoint_path'.")
    ap.add_argument("--data", default="data/expert_data_v4.h5")
    ap.add_argument("--label_key", default="task_labels")
    ap.add_argument("--context_len", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--num_cases", type=int, default=40)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/ablation")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}  sklearn={_HAVE_SKLEARN}")

    images, states, actions, labels, episodes = load_data(args.data, args.label_key)
    N = images.shape[0]
    if N < args.context_len + args.horizon:
        raise ValueError(f"dataset has only {N} frames; need >= "
                         f"{args.context_len + args.horizon}.")

    obstacle_cases, clear_cases = collect_cases(
        labels, episodes, N, args.context_len, args.horizon,
        args.num_cases, args.stride)
    probe_frames = collect_probe_samples(
        labels, episodes, N, args.context_len,
        obstacle_cases + clear_cases, args.horizon)
    print(f"Data: {N} frames. Collected {len(obstacle_cases)} obstacle-cases, "
          f"{len(clear_cases)} clear-cases (target {args.num_cases // 2} each). "
          f"Probe samples: {len(probe_frames)}.")
    if not obstacle_cases:
        print("WARNING: no obstacle-cases found (no clear->obstacle transitions "
              "in windows). D1 obstacle group and D2 will be sparse/empty.")

    results = {}
    for spec in args.model:
        name, cfg_path, ckpt_path = parse_model_spec(spec)
        print(f"\n--- Model '{name}' ---\n    config: {cfg_path}\n    ckpt:   {ckpt_path}")
        wm = build_world_model(cfg_path, ckpt_path, device)
        results[name] = diagnose_model(
            wm, images, states, actions, labels, episodes,
            obstacle_cases, clear_cases, probe_frames,
            args.context_len, args.horizon, device)
        r = results[name]
        flag = " [PROBE WEAK - D2 untrustworthy]" if r["probe_weak"] else ""
        print(f"    probe backend={r['probe_backend']} "
              f"posterior val_sep={r['val_sep']:.3f}{flag}")

    print_tables(results, args.horizon)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "rollout_retention.csv"), results, args.horizon)
    plot_all(results, args.horizon, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
