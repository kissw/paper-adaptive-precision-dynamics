#!/usr/bin/env python3
"""Obstacle-propagation diagnostic for the warp ablation (CARLA-free, offline).

WHY THIS, NOT PSNR/SSIM
-----------------------
README's core finding: the obstacle_head (agent.py) separates obstacle-present
from obstacle-free *posterior* features by ~+276 nats, yet closed-loop avoidance
is 0%. The real bottleneck is not pixel-prediction fidelity but whether the
*action-conditioned prior rollout* (img_step) keeps the obstacle latent alive.
So we do NOT measure decode quality (that is compare_rssm_token_vit_rollout.py).
We measure whether obstacle information survives prior imagination as a function
of horizon. This is the decisive signal for the warp pathway.

METRICS
-------
(M1) Posterior separation [sanity / reproduction]
     Encode context with obs_step (posterior), apply obstacle_head to the
     posterior feature. value = mean(logit | obstacle-present)
                                - mean(logit | obstacle-free).
     Used to confirm the README's +276 nats reproduces (RSSM baseline).

(M2) Prior-rollout separation vs horizon [THE DECISIVE METRIC]
     From the context state, imagine h = 1..H steps with GT actions via img_step.
     At each step apply obstacle_head to the rollout feature and measure the
     present-vs-free logit separation, grouped by the GT obstacle label at that
     future frame.
     Warp hypothesis: M2 COLLAPSES toward 0 with horizon for RSSM (the obstacle
     latent decays in prior imagination), but is RETAINED for warp_token_vit.

(M3) Action sensitivity vs horizon [supporting]
     From the same context, roll out GT actions vs left/right-perturbed actions
     and measure the rollout feature distance per horizon. Warp should keep
     rollouts action-dependent (distance > 0 retained); without it rollouts
     converge (distance -> 0). Needs no obstacle_head, so every model can run it.

GRACEFUL DEGRADATION
--------------------
The obstacle_head module always exists on WorldModel, but it is only *trained*
when beta_obstacle_aux > 0. A checkpoint trained without the aux loss simply
omits obstacle_head.* from its saved state, so after load_state_dict(strict=False)
those keys show up in `missing_keys`. Such models skip M1/M2 and run M3 only.
Which model skips which metric is logged explicitly.

EXAMPLE (real 3-model run; checkpoints live on the server, do not run here):

  ~/.local/bin/uv run python scripts/diagnose_obstacle_propagation.py \
    --model rssm:configs/experiment/task_b_v5.yaml:runs/rssm_.../best.pt \
    --model token_vit:configs/experiment/token_vit.yaml:runs/token_vit_.../best.pt \
    --model warp_token_vit:configs/experiment/task_b_v5_warp.yaml:runs/warp_.../best.pt \
    --data data/expert_data_v4.h5 --num_cases 20 --horizon 15
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

# Repo `src/` on path (mirrors compare_rssm_token_vit_rollout.py layout).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_inference.config import Config  # noqa: E402
from active_inference.agent import WorldModel  # noqa: E402


# ---------------------------------------------------------------------------
# Model building / loading
# ---------------------------------------------------------------------------
def parse_model_spec(spec: str):
    """Parse a --model spec of the form name:config_path:checkpoint_path.

    Paths may themselves contain ':' (split on the first two delimiters only).
    """
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"--model must be 'name:config_path:checkpoint_path', got {spec!r}"
        )
    name = parts[0]
    cfg_path = parts[1]
    ckpt_path = ":".join(parts[2:])
    return name, cfg_path, ckpt_path


def build_and_load(name: str, cfg_path: str, ckpt_path: str, device: torch.device):
    """Build a WorldModel and load its checkpoint (strict=False).

    Returns a dict describing the model and which metrics it supports.
    """
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)

    missing, unexpected, loaded = [], [], False
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "world_model" in ckpt:
            sd = ckpt["world_model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        try:
            result = wm.load_state_dict(sd, strict=False)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            loaded = True
        except RuntimeError as e:
            # strict=False still raises on shape mismatch; surface it clearly
            # rather than silently reshaping anything.
            raise RuntimeError(
                f"[{name}] checkpoint/model parameter shape mismatch while loading "
                f"{ckpt_path}. This usually means the config feat dim differs from "
                f"the trained model. Original error:\n{e}"
            )
    else:
        print(f"[{name}] WARNING: checkpoint not found ({ckpt_path!r}); "
              f"using randomly-initialised weights.")

    wm.eval()

    # obstacle_head always exists on WorldModel; the question is whether the
    # checkpoint actually carried its (trained) weights. If not loaded at all,
    # or if any obstacle_head.* key is missing, treat M1/M2 as unavailable.
    head_weights_missing = any(str(k).startswith("obstacle_head") for k in missing)
    has_head = loaded and not head_weights_missing

    if not loaded:
        print(f"[{name}] no checkpoint loaded -> obstacle_head untrained, "
              f"skipping M1/M2, running M3 only.")
    elif head_weights_missing:
        print(f"[{name}] obstacle_head.* absent from checkpoint (trained without "
              f"aux obstacle loss) -> skipping M1/M2, running M3 only.")
    else:
        print(f"[{name}] obstacle_head loaded -> running M1/M2/M3.")

    if missing:
        print(f"[{name}]   load: {len(missing)} missing keys "
              f"(e.g. {missing[:3]})")
    if unexpected:
        print(f"[{name}]   load: {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")

    # obstacle_head is nn.Sequential(Linear(feat_dim,128), ...). Its input dim
    # must match wm.rssm.get_feat(...) output. We assert this at runtime (below)
    # rather than reshaping anything.
    feat_dim = wm.obstacle_head[0].in_features

    return {
        "name": name,
        "wm": wm,
        "has_head": has_head,
        "feat_dim": feat_dim,
    }


# ---------------------------------------------------------------------------
# Rollout primitives (encoder-agnostic; works for RSSM and TokenViT states).
# Mirrors compare_rssm_token_vit_rollout.py:rollout_model.
# ---------------------------------------------------------------------------
@torch.no_grad()
def context_state(wm, images, states, actions, start, context_len, device):
    """Encode context frames [start, start+context_len) into a posterior state."""
    state = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, actions.shape[-1], device=device)
    for i in range(context_len):
        t = start + i
        img = images[t : t + 1].to(device)
        st = states[t : t + 1].to(device)
        embed = wm.encode_obs(img, st)
        post, _ = wm.rssm.obs_step(state, prev_act, embed)
        state = type(post)(*[x.detach() for x in post])
        prev_act = actions[t : t + 1].to(device)
    return state


@torch.no_grad()
def rollout_states(wm, state, actions, start, context_len, horizon, device):
    """Imagine `horizon` prior steps with GT (or perturbed) actions via img_step.

    The action at index (start+context_len+h-1) transitions to rollout step h,
    matching the convention in compare_rssm_token_vit_rollout.py.
    """
    out = []
    rs = state
    end_ctx = start + context_len
    for h in range(horizon):
        t = end_ctx + h - 1
        act = actions[t : t + 1].to(device)
        rs = wm.rssm.img_step(rs, act)
        out.append(rs)
    return out


@torch.no_grad()
def feat_of(wm, state):
    return wm.rssm.get_feat(state)  # [1, feat_dim]


@torch.no_grad()
def obstacle_logit(wm, state, feat_dim):
    feat = feat_of(wm, state)
    if feat.shape[-1] != feat_dim:
        raise ValueError(
            f"feature dim {feat.shape[-1]} != obstacle_head input dim {feat_dim}; "
            f"refusing to reshape — check the model config."
        )
    return float(wm.obstacle_head(feat).squeeze(-1).item())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(data_path: str, obstacle_key: str):
    """Load images/states/actions/labels. Matches the compare-script format
    (images are float [N, C, H, W]; states [N, S]; actions [N, A])."""
    with h5py.File(data_path, "r") as f:
        for required in ("images", "states", "actions"):
            if required not in f:
                raise KeyError(
                    f"dataset {data_path} is missing required key '{required}'"
                )
        if obstacle_key not in f:
            keys = list(f.keys())
            raise KeyError(
                f"dataset {data_path} has no obstacle label key '{obstacle_key}'. "
                f"This diagnostic requires obstacle labels. Available keys: {keys}. "
                f"Pass the correct --obstacle_key."
            )
        images = torch.tensor(np.array(f["images"]), dtype=torch.float32)
        states = torch.tensor(np.array(f["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(f["actions"]), dtype=torch.float32)
        labels = np.asarray(f[obstacle_key]).reshape(-1).astype(int)
    return images, states, actions, labels


def collect_windows(labels, N, context_len, horizon, num_cases, stride):
    """Select windows, preferring 'obstacle appears in rollout' transitions.

    context = [start, start+context_len); rollout = next `horizon` frames. We
    prefer windows whose context is clear (all labels 0) but whose rollout
    contains an obstacle (some label 1). If too few exist, fall back to plain
    striding. Returns (starts, strategy_str).
    """
    length = context_len + horizon
    all_starts = list(range(0, N - length + 1, stride))

    appearing, others = [], []
    for s in all_starts:
        ctx = labels[s : s + context_len]
        roll = labels[s + context_len : s + length]
        if ctx.sum() == 0 and roll.sum() > 0:
            appearing.append(s)
        else:
            others.append(s)

    if len(appearing) >= num_cases:
        return appearing[:num_cases], (
            f"transition-preferred ({len(appearing)} clear->obstacle windows "
            f"available, using {num_cases})"
        )
    selected = (appearing + others)[:num_cases]
    strategy = (
        f"mixed: {min(len(appearing), num_cases)} transition windows + "
        f"label-grouped fallback (only {len(appearing)} transition windows found)"
    )
    return selected, strategy


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic(models, images, states, actions, labels, starts,
                   context_len, horizon, action_perturb, device):
    """Compute M1/M2/M3 for every model over the selected windows."""
    results = {}
    for m in models:
        results[m["name"]] = {
            "post_logit_present": [],
            "post_logit_free": [],
            "roll_logit": [[] for _ in range(horizon)],  # M2 logits per horizon
            "roll_label": [[] for _ in range(horizon)],   # GT label per horizon
            "act_dist": [[] for _ in range(horizon)],      # M3 distances per horizon
            "has_head": m["has_head"],
        }

    for start in starts:
        post_label = int(labels[start + context_len - 1])
        roll_labels = labels[start + context_len : start + context_len + horizon]

        # Left/right-perturbed action sequences for M3. Channel 0 is steering
        # by CARLA control convention; we perturb only that channel.
        a_left = actions.clone()
        a_right = actions.clone()
        a_left[:, 0] = a_left[:, 0] - action_perturb
        a_right[:, 0] = a_right[:, 0] + action_perturb

        for m in models:
            wm = m["wm"]
            r = results[m["name"]]

            state = context_state(wm, images, states, actions, start,
                                  context_len, device)

            # M1: posterior separation
            if m["has_head"]:
                logit = obstacle_logit(wm, state, m["feat_dim"])
                (r["post_logit_present"] if post_label == 1
                 else r["post_logit_free"]).append(logit)

            # M2: prior-rollout separation (GT actions)
            gt_states = rollout_states(wm, state, actions, start, context_len,
                                       horizon, device)
            if m["has_head"]:
                for h, s in enumerate(gt_states):
                    r["roll_logit"][h].append(obstacle_logit(wm, s, m["feat_dim"]))
                    r["roll_label"][h].append(int(roll_labels[h]))

            # M3: action sensitivity (GT vs perturbed feature distance)
            left_states = rollout_states(wm, state, a_left, start, context_len,
                                         horizon, device)
            right_states = rollout_states(wm, state, a_right, start, context_len,
                                          horizon, device)
            for h in range(horizon):
                f_gt = feat_of(wm, gt_states[h])
                f_l = feat_of(wm, left_states[h])
                f_r = feat_of(wm, right_states[h])
                d = 0.5 * (torch.norm(f_gt - f_l).item()
                           + torch.norm(f_gt - f_r).item())
                r["act_dist"][h].append(d)

    return results


def separation(present, free):
    if not present or not free:
        return None
    return float(np.mean(present) - np.mean(free))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(out_csv, models, results, horizon):
    rows = []
    for m in models:
        name = m["name"]
        r = results[name]
        if m["has_head"]:
            sep = separation(r["post_logit_present"], r["post_logit_free"])
            rows.append((name, "posterior_separation", 0,
                         "" if sep is None else f"{sep:.6f}",
                         len(r["post_logit_present"]), len(r["post_logit_free"])))
            for h in range(horizon):
                logits = np.asarray(r["roll_logit"][h])
                labs = np.asarray(r["roll_label"][h])
                present = logits[labs == 1].tolist()
                free = logits[labs == 0].tolist()
                sep = separation(present, free)
                rows.append((name, "prior_rollout_separation", h + 1,
                             "" if sep is None else f"{sep:.6f}",
                             len(present), len(free)))
        for h in range(horizon):
            dists = r["act_dist"][h]
            val = float(np.mean(dists)) if dists else None
            rows.append((name, "action_sensitivity", h + 1,
                         "" if val is None else f"{val:.6f}", len(dists), 0))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "metric", "horizon_step", "value",
                    "n_present", "n_free"])
        w.writerows(rows)
    print(f"\nWrote CSV: {out_csv}")
    return rows


def print_m2_table(models, results, horizon):
    head_models = [m for m in models if m["has_head"]]
    if not head_models:
        print("\n[M2] no model has a trained obstacle_head; skipping M2 table.")
        return
    print("\n=== M2: prior-rollout obstacle separation (nats) vs horizon ===")
    header = "h   " + "  ".join(f"{m['name']:>16}" for m in head_models)
    print(header)
    print("-" * len(header))
    for h in range(horizon):
        cells = []
        for m in head_models:
            r = results[m["name"]]
            logits = np.asarray(r["roll_logit"][h])
            labs = np.asarray(r["roll_label"][h])
            sep = separation(logits[labs == 1].tolist(),
                             logits[labs == 0].tolist())
            cells.append(f"{'n/a':>16}" if sep is None else f"{sep:>16.3f}")
        print(f"{h + 1:<3} " + "  ".join(cells))


def plot_m2(models, results, horizon, out_png):
    plt.figure(figsize=(7, 5))
    plotted = False
    for m in models:
        if not m["has_head"]:
            continue
        r = results[m["name"]]
        xs, ys = [], []
        for h in range(horizon):
            logits = np.asarray(r["roll_logit"][h])
            labs = np.asarray(r["roll_label"][h])
            sep = separation(logits[labs == 1].tolist(),
                             logits[labs == 0].tolist())
            if sep is not None:
                xs.append(h + 1)
                ys.append(sep)
        if xs:
            plt.plot(xs, ys, marker="o", label=m["name"])
            plotted = True
    plt.axhline(0.0, color="gray", lw=0.8, ls="--")
    plt.xlabel("rollout horizon step")
    plt.ylabel("obstacle separation (present - free) [nats]")
    plt.title("M2: prior-rollout obstacle separation vs horizon\n"
              "(warp hypothesis: retained for warp, collapses for RSSM)")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote figure: {out_png}")


def plot_m3(models, results, horizon, out_png):
    plt.figure(figsize=(7, 5))
    for m in models:
        r = results[m["name"]]
        xs = list(range(1, horizon + 1))
        ys = [float(np.mean(r["act_dist"][h])) if r["act_dist"][h] else np.nan
              for h in range(horizon)]
        plt.plot(xs, ys, marker="o", label=m["name"])
    plt.xlabel("rollout horizon step")
    plt.ylabel("GT-vs-perturbed feature distance (L2)")
    plt.title("M3: action sensitivity vs horizon\n"
              "(warp hypothesis: retained for warp, decays for RSSM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote figure: {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", action="append", required=True,
                    help="Repeatable. Format 'name:config_path:checkpoint_path'.")
    ap.add_argument("--data", required=True,
                    help="HDF5 dataset with obstacle labels.")
    ap.add_argument("--obstacle_key", default="obstacle_labels")
    ap.add_argument("--context_len", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--num_cases", type=int, default=20)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--action_perturb", type=float, default=0.5,
                    help="Steering (action channel 0) perturbation for M3.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="outputs/ablation")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )
    print(f"Device: {device}")

    models = []
    for spec in args.model:
        name, cfg_path, ckpt_path = parse_model_spec(spec)
        print(f"\n--- Loading model '{name}' ---")
        print(f"    config:     {cfg_path}")
        print(f"    checkpoint: {ckpt_path}")
        models.append(build_and_load(name, cfg_path, ckpt_path, device))

    images, states, actions, labels = load_data(args.data, args.obstacle_key)
    N = images.shape[0]
    length = args.context_len + args.horizon
    if N < length:
        raise ValueError(
            f"dataset has only {N} frames; need at least context_len+horizon="
            f"{length}."
        )
    starts, strategy = collect_windows(labels, N, args.context_len, args.horizon,
                                       args.num_cases, args.stride)
    if not starts:
        raise ValueError("no valid windows selected; check stride/num_cases/N.")
    print(f"\nData: {N} frames. Selected {len(starts)} windows. "
          f"Strategy: {strategy}")

    results = run_diagnostic(models, images, states, actions, labels, starts,
                             args.context_len, args.horizon, args.action_perturb,
                             device)

    # M1 summary
    print("\n=== M1: posterior obstacle separation (nats) ===")
    for m in models:
        if not m["has_head"]:
            print(f"  {m['name']:>16}: skipped (no trained obstacle_head)")
            continue
        r = results[m["name"]]
        sep = separation(r["post_logit_present"], r["post_logit_free"])
        np_, nf_ = len(r["post_logit_present"]), len(r["post_logit_free"])
        if sep is None:
            print(f"  {m['name']:>16}: n/a (need both present & free posterior "
                  f"cases; present={np_} free={nf_})")
        else:
            print(f"  {m['name']:>16}: {sep:+.3f}  (present={np_} free={nf_})")

    print_m2_table(models, results, args.horizon)

    out_csv = os.path.join(args.out_dir, "obstacle_propagation.csv")
    write_csv(out_csv, models, results, args.horizon)
    plot_m2(models, results, args.horizon,
            os.path.join(args.out_dir, "m2_prior_rollout_separation.png"))
    plot_m3(models, results, args.horizon,
            os.path.join(args.out_dir, "m3_action_sensitivity.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
