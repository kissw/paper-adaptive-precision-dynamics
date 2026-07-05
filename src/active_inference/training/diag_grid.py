"""Diagnostic rollout grid for world-model training.

Produces a 5-row × len(horizons)-col image grid to visually track how the
world model's prior rollout and decoder evolve as training progresses.

Rows:
    GT        — ground-truth future frames (crop applied)
    recon     — encode each future frame (with context) → decode
    rollout   — prior rollout from start posterior using GT actions → decode
    steer=+H  — prior rollout with fixed action [+steer_mag, nominal_accel]
    steer=-H  — prior rollout with fixed action [-steer_mag, nominal_accel]

CARLA steer convention: +steer = right turn, -steer = left turn.

All rollouts are deterministic (stochastic sample replaced by its mean) so
successive grids reflect weight changes, not sampling noise.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal


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
    return type(state)(deter=state.deter, stoch=state.mean,
                       mean=state.mean, std=state.std)


def _to_display(img: Tensor) -> np.ndarray:
    """(3,H,W) float in ~[-0.5,0.5] → (H,W,3) in [0,1]."""
    arr = (img.detach().float().cpu() + 0.5).clamp(0.0, 1.0)
    return arr.permute(1, 2, 0).numpy()


@torch.no_grad()
def _compute_diag_rows(
    wm,
    images: Tensor,
    states: Tensor,
    actions: Tensor,
    start_idx: int,
    horizons: list[int],
    context_len: int,
    steer_mag: float,
    device: torch.device,
) -> tuple[list[int], list[tuple[str, dict[int, Tensor]]]]:
    T_total = images.shape[0]
    h_max = max(horizons)
    if start_idx + h_max >= T_total:
        horizons = [h for h in horizons if start_idx + h < T_total]
        if not horizons:
            return [], []
        h_max = max(horizons)

    action_dim = actions.shape[-1]

    ctx_start = max(0, start_idx - context_len + 1)
    state = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, action_dim, device=device)
    with _det_normal():
        for t in range(ctx_start, start_idx + 1):
            img = wm.preprocess_image(images[t:t+1].to(device))
            st = states[t:t+1].to(device)
            embed = wm.encoder(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            state = _make_det(post)
            prev_act = actions[t:t+1].to(device)
    start_post = state

    nominal_accel = float(actions[start_idx, 1]) if action_dim >= 2 else 0.3

    gt_row, recon_row = {}, {}
    for h in horizons:
        gt_row[h] = wm.preprocess_image(
            images[start_idx + h:start_idx + h + 1].to(device)
        ).squeeze(0)

    with _det_normal():
        for h in horizons:
            tgt = start_idx + h
            cs = max(0, tgt - context_len + 1)
            s = wm.rssm.initial(1, device)
            pa = torch.zeros(1, action_dim, device=device)
            for t in range(cs, tgt + 1):
                img = wm.preprocess_image(images[t:t+1].to(device))
                st = states[t:t+1].to(device)
                embed = wm.encoder(img, st)
                p, _ = wm.rssm.obs_step(s, pa, embed)
                s = _make_det(p)
                pa = actions[t:t+1].to(device)
            recon_row[h] = wm.decode_obs(s).squeeze(0)

    def _rollout(action_fn):
        out = {}
        with _det_normal():
            s = start_post
            for h in range(1, h_max + 1):
                act = action_fn(h).to(device)
                s = _make_det(wm.rssm.img_step(s, act))
                if h in horizons:
                    out[h] = wm.decode_obs(s).squeeze(0)
        return out

    rollout_row = _rollout(lambda h: actions[start_idx + h:start_idx + h + 1])
    left_act = torch.tensor([[+steer_mag, nominal_accel]], dtype=torch.float32)
    right_act = torch.tensor([[-steer_mag, nominal_accel]], dtype=torch.float32)
    left_row = _rollout(lambda h: left_act)
    right_row = _rollout(lambda h: right_act)

    rows = [
        ("GT", gt_row),
        ("recon", recon_row),
        ("rollout", rollout_row),
        (f"steer=+{steer_mag:g} (R)", left_row),
        (f"steer=-{steer_mag:g} (L)", right_row),
    ]
    return horizons, rows


@torch.no_grad()
def make_diag_grid(
    wm,
    images: Tensor,        # (T_total, 3, H, W) raw frames for ONE episode window
    states: Tensor,        # (T_total, state_dim)
    actions: Tensor,       # (T_total, action_dim)
    start_idx: int,        # index within the window to start rollout from
    output_path: str | Path,
    horizons: list[int] | None = None,
    context_len: int = 5,
    steer_mag: float = 0.2,
    device: torch.device | None = None,
    title: str = "",
) -> bool:
    """Render and save the 5-row diagnostic grid. Returns True on success."""
    horizons = horizons or [1, 5, 10, 15]
    if device is None:
        device = next(wm.parameters()).device

    T_total = images.shape[0]
    h_max = max(horizons)
    if start_idx + h_max >= T_total:
        # Not enough future frames; clip horizons that fit
        horizons = [h for h in horizons if start_idx + h < T_total]
        if not horizons:
            return False
        h_max = max(horizons)

    action_dim = actions.shape[-1]

    # ── Build start posterior via context obs_steps ───────────────────────
    ctx_start = max(0, start_idx - context_len + 1)
    state = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, action_dim, device=device)
    with _det_normal():
        for t in range(ctx_start, start_idx + 1):
            img = wm.preprocess_image(images[t:t+1].to(device))
            st = states[t:t+1].to(device)
            embed = wm.encoder(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            state = _make_det(post)
            prev_act = actions[t:t+1].to(device)
    start_post = state

    nominal_accel = float(actions[start_idx, 1]) if action_dim >= 2 else 0.3

    # ── Row data ──────────────────────────────────────────────────────────
    gt_row, recon_row, rollout_row = {}, {}, {}
    left_row, right_row = {}, {}

    # GT row (preprocessed for crop consistency)
    for h in horizons:
        gt_row[h] = wm.preprocess_image(
            images[start_idx + h:start_idx + h + 1].to(device)
        ).squeeze(0)

    # recon row: encode each target frame with its own short context
    with _det_normal():
        for h in horizons:
            tgt = start_idx + h
            cs = max(0, tgt - context_len + 1)
            s = wm.rssm.initial(1, device)
            pa = torch.zeros(1, action_dim, device=device)
            for t in range(cs, tgt + 1):
                img = wm.preprocess_image(images[t:t+1].to(device))
                st = states[t:t+1].to(device)
                embed = wm.encoder(img, st)
                p, _ = wm.rssm.obs_step(s, pa, embed)
                s = _make_det(p)
                pa = actions[t:t+1].to(device)
            recon_row[h] = wm.decode_obs(s).squeeze(0)

    def _rollout(action_fn):
        out = {}
        with _det_normal():
            s = start_post
            for h in range(1, h_max + 1):
                act = action_fn(h).to(device)
                s = _make_det(wm.rssm.img_step(s, act))
                if h in horizons:
                    out[h] = wm.decode_obs(s).squeeze(0)
        return out

    # rollout row: GT actions
    rollout_row = _rollout(
        lambda h: actions[start_idx + h:start_idx + h + 1]
    )
    # fixed-steer holds
    left_act = torch.tensor([[+steer_mag, nominal_accel]], dtype=torch.float32)
    right_act = torch.tensor([[-steer_mag, nominal_accel]], dtype=torch.float32)
    left_row = _rollout(lambda h: left_act)
    right_row = _rollout(lambda h: right_act)

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = [
        ("GT",                gt_row),
        ("recon",             recon_row),
        ("rollout",           rollout_row),
        (f"steer=+{steer_mag:g} (R)", left_row),
        (f"steer=-{steer_mag:g} (L)", right_row),
    ]
    n_rows, n_cols = len(rows), len(horizons)
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=((n_cols + 0.7) * 1.4, n_rows * 1.5),
        gridspec_kw={"width_ratios": [0.7] + [1.0] * n_cols},
        dpi=120,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, (label, data) in enumerate(rows):
        lax = axes[r, 0]
        lax.axis("off")
        lax.text(0.95, 0.5, label, transform=lax.transAxes,
                 fontsize=8, ha="right", va="center",
                 fontweight="bold" if label == "GT" else "normal")
        for c, h in enumerate(horizons):
            ax = axes[r, c + 1]
            ax.axis("off")
            if h in data:
                ax.imshow(_to_display(data[h]), interpolation="nearest")
            else:
                ax.set_facecolor("#222")
            if r == 0:
                ax.set_title(f"t+{h}", fontsize=8, pad=3, fontweight="bold")

    fig.suptitle(title or "Diagnostic rollout grid", fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


@torch.no_grad()
def make_multi_diag_grid(
    wm,
    samples: list[tuple[Tensor, Tensor, Tensor, int]],
    output_path: str | Path,
    horizons: list[int] | None = None,
    context_len: int = 5,
    steer_mag: float = 0.2,
    device: torch.device | None = None,
    title: str = "",
) -> bool:
    """Render one 5-row grid with columns grouped by sample and horizon."""
    horizons = horizons or [1, 5, 10, 15]
    if not samples:
        return False
    if device is None:
        device = next(wm.parameters()).device

    rendered = []
    row_labels = None
    for sample_idx, (images, states, actions, start_idx) in enumerate(samples, start=1):
        hs, rows = _compute_diag_rows(
            wm, images, states, actions, start_idx,
            horizons, context_len, steer_mag, device,
        )
        if not hs:
            continue
        rendered.append((sample_idx, hs, rows))
        if row_labels is None:
            row_labels = [label for label, _ in rows]

    if not rendered or row_labels is None:
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    n_rows = len(row_labels)
    n_cols = sum(len(hs) for _, hs, _ in rendered)
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=(max(2.2 * (n_cols + 0.7), 6.0), 2.2 * n_rows),
        gridspec_kw={"width_ratios": [0.8] + [1.0] * n_cols},
        dpi=120,
        squeeze=False,
    )

    for r, label in enumerate(row_labels):
        lax = axes[r, 0]
        lax.axis("off")
        lax.text(0.95, 0.5, label, transform=lax.transAxes,
                 fontsize=9, ha="right", va="center",
                 fontweight="bold" if label == "GT" else "normal")

    c = 1
    for sample_idx, hs, rows in rendered:
        row_data = {label: data for label, data in rows}
        for h in hs:
            for r, label in enumerate(row_labels):
                ax = axes[r, c]
                ax.axis("off")
                data = row_data[label]
                if h in data:
                    ax.imshow(_to_display(data[h]), interpolation="nearest")
                else:
                    ax.set_facecolor("#222")
                if r == 0:
                    ax.set_title(
                        f"S{sample_idx} t+{h}",
                        fontsize=8,
                        pad=3,
                        fontweight="bold",
                    )
            c += 1

    fig.suptitle(title or "Diagnostic rollout grid", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.35, h_pad=0.2, w_pad=0.15)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True
