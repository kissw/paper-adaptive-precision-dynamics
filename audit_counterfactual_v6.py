"""Audit a v6 counterfactual smoke HDF5 (metadata-only, no image load).

Verifies that the improved collection (short-hold cf_action + expert recovery,
extended branch_horizon, reduced context_len) behaves as intended.

Checks
------
1. branch action recovery: action[step 0..hold-1] == cf_action,
   action[step hold..] == expert/nominal action.
2. branch_horizon: max cf_branch_step.
3. context_len: number of cf_is_context==True frames per episode.
4. cte accumulation: distribution of |crosstrack| over branch_step,
   to confirm late-branch frames are no longer drifting unrealistically.
5. episode structure: frames per episode_id.

Usage
-----
    python audit_counterfactual_v6.py --input data/counterfactual_v6_smoke.h5 \
        --output_dir outputs/audit_v6 --expected_hold 2 \
        --expected_branch_horizon 20 --expected_context_len 30
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_meta(path):
    """Load only metadata arrays (skip images for speed/memory)."""
    keys = [
        "episode_ids", "actions", "states",
        "cf_is_context", "cf_is_branch", "cf_branch_step",
        "cf_sampled_steer", "cf_steer_bias", "cf_anchor_id", "cf_branch_id",
        "cf_max_abs_cte_delta_diag", "cf_max_abs_cte_delta_10",
        "expert_actions",
    ]
    out = {}
    with h5py.File(path, "r") as f:
        present = list(f.keys())
        for k in keys:
            if k in f:
                out[k] = f[k][:]
        out["_attrs"] = dict(f.attrs)
        out["_present_keys"] = present
    return out


def check_branch_action_recovery(d, expected_hold):
    """For each branch, verify action holds cf_action then returns to expert."""
    if "cf_is_branch" not in d or "actions" not in d:
        return "SKIP: missing cf_is_branch or actions"

    is_branch = d["cf_is_branch"].astype(bool)
    branch_step = d.get("cf_branch_step")
    actions = d["actions"]
    sampled_steer = d.get("cf_sampled_steer")
    expert_actions = d.get("expert_actions")

    if branch_step is None or sampled_steer is None:
        return "SKIP: missing cf_branch_step or cf_sampled_steer"

    # branch_step convention: 1-indexed in collector (i - context_len + 1)
    # so step 1..hold = cf_action, step hold+1.. = expert
    steer_col = actions[:, 0]
    bidx = np.where(is_branch)[0]
    if len(bidx) == 0:
        return "FAIL: no branch frames found"

    hold_mask = is_branch & (branch_step <= expected_hold)
    post_mask = is_branch & (branch_step > expected_hold)

    # During hold: steer should match sampled_steer
    hold_diff = np.abs(steer_col[hold_mask] - sampled_steer[hold_mask])
    hold_ok = np.mean(hold_diff < 1e-3) if hold_mask.sum() > 0 else float("nan")

    # After hold: steer should match expert action steer (not sampled_steer)
    if expert_actions is not None:
        post_diff_expert = np.abs(
            steer_col[post_mask] - expert_actions[post_mask, 0]
        )
        post_ok_expert = (
            np.mean(post_diff_expert < 1e-3) if post_mask.sum() > 0
            else float("nan")
        )
        post_diff_sampled = np.abs(
            steer_col[post_mask] - sampled_steer[post_mask]
        )
        post_still_cf = (
            np.mean(post_diff_sampled < 1e-3) if post_mask.sum() > 0
            else float("nan")
        )
    else:
        post_ok_expert = float("nan")
        post_still_cf = float("nan")

    lines = [
        f"  hold frames (step<={expected_hold}): {int(hold_mask.sum())}",
        f"    fraction matching cf_action steer : {hold_ok:.3f}  (expect ~1.0)",
        f"  post-hold frames (step>{expected_hold}): {int(post_mask.sum())}",
        f"    fraction matching EXPERT steer    : {post_ok_expert:.3f}  (expect ~1.0)",
        f"    fraction STILL at cf_action steer : {post_still_cf:.3f}  (expect ~0.0)",
    ]
    verdict = "PASS" if (hold_ok > 0.95 and post_ok_expert > 0.95) else "CHECK"
    return f"[{verdict}]\n" + "\n".join(lines)


def check_branch_horizon(d, expected):
    if "cf_branch_step" not in d:
        return "SKIP: missing cf_branch_step"
    bs = d["cf_branch_step"]
    is_branch = d["cf_is_branch"].astype(bool) if "cf_is_branch" in d else (bs >= 1)
    max_step = int(bs[is_branch].max()) if is_branch.sum() > 0 else 0
    verdict = "PASS" if max_step == expected else "CHECK"
    return f"[{verdict}] max cf_branch_step = {max_step}  (expect {expected})"


def check_context_len(d, expected):
    if "cf_is_context" not in d or "episode_ids" not in d:
        return "SKIP: missing cf_is_context or episode_ids"
    is_ctx = d["cf_is_context"].astype(bool)
    eids = d["episode_ids"]
    per_ep = []
    for e in np.unique(eids):
        per_ep.append(int(is_ctx[eids == e].sum()))
    per_ep = np.array(per_ep)
    verdict = "PASS" if np.all(per_ep == expected) else "CHECK"
    return (f"[{verdict}] context frames/episode: "
            f"min={per_ep.min()} median={int(np.median(per_ep))} "
            f"max={per_ep.max()}  (expect {expected})")


def check_episode_structure(d, expected_total):
    eids = d["episode_ids"]
    per_ep = np.array([int((eids == e).sum()) for e in np.unique(eids)])
    verdict = "PASS" if np.all(per_ep == expected_total) else "CHECK"
    return (f"[{verdict}] frames/episode: min={per_ep.min()} "
            f"median={int(np.median(per_ep))} max={per_ep.max()} "
            f"n_episodes={len(per_ep)}  (expect {expected_total})")


def check_cte_accumulation(d, out_dir, expected_hold):
    """Compare |crosstrack| over branch_step. Late steps should not blow up."""
    if "cf_branch_step" not in d or "states" not in d:
        return "SKIP: missing cf_branch_step or states"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "SKIP: matplotlib not available"

    bs = d["cf_branch_step"]
    is_branch = d["cf_is_branch"].astype(bool) if "cf_is_branch" in d else (bs >= 1)
    cte = np.abs(d["states"][:, 3])  # crosstrack_error is index 3

    steps = sorted(np.unique(bs[is_branch]).tolist())
    means, p90s = [], []
    for s in steps:
        m = is_branch & (bs == s)
        vals = cte[m]
        means.append(float(vals.mean()) if len(vals) else np.nan)
        p90s.append(float(np.percentile(vals, 90)) if len(vals) else np.nan)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, means, "o-", label="mean |crosstrack|")
    ax.plot(steps, p90s, "s--", label="p90 |crosstrack|")
    ax.axvline(expected_hold + 0.5, color="red", ls=":",
               label=f"hold ends (step {expected_hold})")
    ax.set_xlabel("cf_branch_step")
    ax.set_ylabel("|crosstrack_error|")
    ax.set_title("Crosstrack accumulation over branch (should flatten after hold)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_dir) / "cte_accumulation.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    # Heuristic: does mean keep growing monotonically through the end?
    post_means = [means[i] for i, s in enumerate(steps) if s > expected_hold]
    growing = (len(post_means) >= 2
               and post_means[-1] > post_means[0] * 1.5)
    verdict = "CHECK (still growing post-hold)" if growing else "PASS (flattens)"
    return (f"[{verdict}] saved plot {out_path}\n"
            f"  mean |cte| at first/last branch step: "
            f"{means[0]:.3f} / {means[-1]:.3f}")


def check_recovery(d, expected_hold):
    """Verify closed-loop recovery: |crosstrack| should DECREASE after hold,
    and check whether brake (negative accel) entered the action distribution."""
    if "cf_branch_step" not in d or "states" not in d:
        return "SKIP: missing cf_branch_step or states"

    bs = d["cf_branch_step"]
    is_branch = d["cf_is_branch"].astype(bool) if "cf_is_branch" in d else (bs >= 1)
    cte = np.abs(d["states"][:, 3])
    actions = d.get("actions")

    steps = sorted(np.unique(bs[is_branch]).tolist())
    means = {}
    for s in steps:
        m = is_branch & (bs == s)
        means[s] = float(cte[m].mean()) if m.sum() else np.nan

    # crosstrack at hold-end vs branch-end
    post_steps = [s for s in steps if s > expected_hold]
    if len(post_steps) < 2:
        return "SKIP: not enough post-hold steps"
    cte_at_hold_end = means[post_steps[0]]
    cte_peak = max(means[s] for s in post_steps)
    cte_at_branch_end = means[post_steps[-1]]

    # Recovery success: end < peak (came back down) AND end <= hold_end*1.1
    recovered = (cte_at_branch_end < cte_peak * 0.95) or (
        cte_at_branch_end <= cte_at_hold_end * 1.1
    )

    # Brake usage in recovery region (accel < 0 means braking)
    brake_note = ""
    if actions is not None and actions.shape[1] >= 2:
        post_mask = is_branch & (bs > expected_hold)
        accel = actions[post_mask, 1]
        brake_frac = float(np.mean(accel < 0.0)) if post_mask.sum() else 0.0
        brake_note = (f"\n  recovery-region brake fraction (accel<0): "
                      f"{brake_frac:.3f}  (was ~0 in original fixed-throttle data)")

    verdict = "PASS (recovers)" if recovered else "FAIL (no recovery)"
    return (f"[{verdict}]\n"
            f"  mean |cte| hold-end(step {post_steps[0]}) = {cte_at_hold_end:.3f}\n"
            f"  mean |cte| peak               = {cte_peak:.3f}\n"
            f"  mean |cte| branch-end(step {post_steps[-1]}) = {cte_at_branch_end:.3f}"
            f"{brake_note}\n"
            f"  -> 복귀 성공이면 branch-end가 peak보다 확실히 작아야 함")


def check_colored_noise_branch(d, expected_hold=None):
    """Verify colored-noise branch actions: per-branch temporal diversity,
    step-to-step smoothness, and branch-to-branch divergence."""
    if "actions" not in d or "cf_is_branch" not in d:
        return "SKIP: missing actions or cf_is_branch"
    actions = d["actions"]
    is_branch = d["cf_is_branch"].astype(bool)
    eids = d["episode_ids"]
    cte = np.abs(d["states"][:, 3]) if "states" in d else None
    steer_bias = d.get("cf_steer_bias")

    # (a) per-branch temporal diversity of steer
    divers = []
    # (b) step-to-step smoothness (mean abs diff of steer within branch)
    smooth = []
    for e in np.unique(eids):
        m = (eids == e) & is_branch
        if m.sum() < 2:
            continue
        steer = actions[m, 0]
        divers.append(float(steer.std()))
        smooth.append(float(np.mean(np.abs(np.diff(steer)))))
    divers = np.array(divers)
    smooth = np.array(smooth)

    # (c) branch divergence: correlation between steer_bias and final cte
    div_note = ""
    if steer_bias is not None and cte is not None:
        per_branch_bias, per_branch_cte = [], []
        for e in np.unique(eids):
            m = (eids == e) & is_branch
            if m.sum() == 0:
                continue
            per_branch_bias.append(float(steer_bias[m][0]))
            per_branch_cte.append(float(cte[m][-1]))  # final cte
        if len(per_branch_bias) > 3:
            corr = float(np.corrcoef(np.abs(per_branch_bias),
                                     per_branch_cte)[0, 1])
            div_note = (f"\n  |steer_bias| vs final|cte| corr: {corr:.3f}  "
                        f"(>0.3 이면 bias가 divergence 유발 = sensitivity 확보)")

    # (d) bias vs actual_mean_steer Spearman rank correlation
    spearman_note = ""
    if steer_bias is not None:
        per_branch_bias2, per_branch_mean_steer = [], []
        for e in np.unique(eids):
            m = (eids == e) & is_branch
            if m.sum() == 0:
                continue
            per_branch_bias2.append(float(steer_bias[m][0]))
            per_branch_mean_steer.append(float(actions[m, 0].mean()))
        if len(per_branch_bias2) > 3:
            b = np.array(per_branch_bias2)
            s = np.array(per_branch_mean_steer)
            # Spearman: rank-based Pearson (no scipy dependency)
            rb = np.argsort(np.argsort(b)).astype(float)
            rs = np.argsort(np.argsort(s)).astype(float)
            sp_corr = float(np.corrcoef(rb, rs)[0, 1])
            spearman_note = (
                f"\n  bias vs actual mean steer Spearman corr: {sp_corr:.3f}  "
                f"(1.0에 가까우면 bias가 실제 조향 지배 = 체계적 coverage 유지, "
                f"낮으면 noise가 bias를 뒤집음 = coverage 약함)"
            )

    lines = [
        f"  per-branch steer std (temporal diversity): "
        f"mean={divers.mean():.3f}  (>0.02 expect; 고정이면 0)",
        f"  step-to-step |dsteer| (smoothness): "
        f"mean={smooth.mean():.3f}  (작을수록 smooth; white noise면 큼)",
    ]
    diversity_ok = divers.mean() > 0.02
    verdict = "PASS" if diversity_ok else "CHECK (sequence가 고정에 가까움)"
    return f"[{verdict}]\n" + "\n".join(lines) + div_note + spearman_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="outputs/audit_v6")
    ap.add_argument("--expected_hold", type=int, default=2)
    ap.add_argument("--expected_branch_horizon", type=int, default=20)
    ap.add_argument("--expected_context_len", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_meta(args.input)
    expected_total = args.expected_context_len + args.expected_branch_horizon

    print("=" * 64)
    print(f"AUDIT: {args.input}")
    print("=" * 64)
    print(f"present keys: {d['_present_keys']}")
    print(f"attrs: {d['_attrs']}")
    print()
    print("--- 1. branch action recovery (hold then expert) ---")
    print(check_branch_action_recovery(d, args.expected_hold))
    print()
    print("--- 2. branch_horizon ---")
    print(check_branch_horizon(d, args.expected_branch_horizon))
    print()
    print("--- 3. context_len ---")
    print(check_context_len(d, args.expected_context_len))
    print()
    print("--- 4. episode structure ---")
    print(check_episode_structure(d, expected_total))
    print()
    print("--- 5. crosstrack accumulation ---")
    print(check_cte_accumulation(d, out_dir, args.expected_hold))
    print()
    print("--- 6. closed-loop recovery (only for recovery-mode data) ---")
    print(check_recovery(d, args.expected_hold))
    print()
    print("--- 7. colored-noise branch (diversity / smoothness / divergence) ---")
    print(check_colored_noise_branch(d))
    print()
    print("=" * 64)
    print("Done. Review the [CHECK] items and the cte_accumulation.png plot.")


if __name__ == "__main__":
    main()