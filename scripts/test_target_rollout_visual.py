"""Standalone visual test for target_rollout transition checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from active_inference.agent import DeepAIFAgent
from active_inference.config import Config
from active_inference.utils.bbox import bbox_xyxy_to_model_space


def _parse_horizons(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    return out or None


def _default_horizons(cfg: Config) -> list[int]:
    rollout_horizon = int(getattr(cfg.training, "rollout_horizon", 1))
    decode = getattr(cfg.training, "rollout_decode_horizons", None)
    if decode:
        horizons = [int(h) for h in decode]
    else:
        horizons = [1]
    horizons = [h for h in horizons if 1 <= h <= rollout_horizon]
    return horizons or [1]


def _det_state(state):
    if hasattr(state, "token_mean"):
        return type(state)(
            deter=state.deter,
            stoch=state.token_mean,
            mean=state.mean,
            std=state.std,
            token_mean=state.token_mean,
            token_std=state.token_std,
        )
    return type(state)(
        deter=state.deter,
        stoch=state.mean,
        mean=state.mean,
        std=state.std,
    )


def _slice_state(state, b: int):
    return type(state)(*[x[b:b + 1] for x in state])


def _to_display(img: torch.Tensor) -> torch.Tensor:
    return (img.detach().float().cpu() + 0.5).clamp(0.0, 1.0).permute(1, 2, 0)


def _model_bbox_valid(bbox, cfg: Config, image_size: int) -> bool:
    model_bbox = bbox_xyxy_to_model_space(
        torch.as_tensor(bbox, dtype=torch.float32),
        image_h=image_size,
        image_w=image_size,
        crop_road=bool(getattr(cfg.encoder, "crop_road", False)),
        keep_bottom_frac=float(getattr(cfg.encoder, "keep_bottom_frac", 0.6)),
        coord_space=getattr(cfg.training, "bbox_coord_space", "raw"),
        min_area=float(getattr(cfg.training, "bbox_min_area", 1.0)),
    )
    return bool(torch.isfinite(model_bbox).all())


def _select_windows(
    h5_path: str,
    cfg: Config,
    num_samples: int,
    context_idx: int,
    horizons: list[int],
    obstacle_visible: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import h5py
    import numpy as np

    seq_len = int(cfg.training.seq_len)
    image_size = int(getattr(cfg.encoder, "image_size", 64))
    if hasattr(cfg, "token_vit"):
        image_size = int(getattr(cfg.token_vit, "image_size", image_size))

    with h5py.File(h5_path, "r") as f:
        episode_ids = f["episode_ids"][:]
        bboxes = f["obstacle_bbox"][:] if "obstacle_bbox" in f else None
        n = len(episode_ids)

        def can_use_start(start: int) -> bool:
            end = start + seq_len
            if start < 0 or end > n:
                return False
            if episode_ids[start] != episode_ids[end - 1]:
                return False
            return context_idx + max(horizons) < seq_len

        def obstacle_ok(start: int) -> bool:
            if bboxes is None:
                return False
            for h in horizons:
                idx = start + context_idx + h
                if idx < n and episode_ids[idx] == episode_ids[start]:
                    if _model_bbox_valid(bboxes[idx], cfg, image_size):
                        return True
            return False

        valid_starts = [
            s for s in range(0, n - seq_len + 1) if can_use_start(s)
        ]
        if not valid_starts:
            raise RuntimeError(
                f"No valid windows of seq_len={seq_len} found in {h5_path}"
            )

        preferred = (
            [s for s in valid_starts if obstacle_ok(s)]
            if obstacle_visible else []
        )

        def spread(starts: list[int], need: int) -> list[int]:
            if need <= 0 or not starts:
                return []
            if len(starts) <= need:
                return starts
            pos = np.linspace(0, len(starts) - 1, need).round().astype(int)
            return [starts[int(p)] for p in pos]

        selected: list[int] = []
        selected.extend(spread(preferred, num_samples))
        seen = set(selected)
        selected.extend(
            s for s in spread(valid_starts, num_samples * 2)
            if s not in seen
        )
        selected = selected[:num_samples]

        images = []
        states = []
        actions = []
        for start in selected:
            end = start + seq_len
            images.append(torch.from_numpy(f["images"][start:end]).float())
            states.append(torch.from_numpy(f["states"][start:end]).float())
            actions.append(torch.from_numpy(f["actions"][start:end]).float())

    return torch.stack(images, dim=0), torch.stack(states, dim=0), torch.stack(actions, dim=0)


@torch.no_grad()
def _posterior_sequence(wm, images, states, actions, device):
    bsz, timesteps = images.shape[:2]
    prev = wm.rssm.initial(bsz, device)
    posts = []
    for t in range(timesteps):
        img_t = wm.preprocess_image(images[:, t].to(device))
        st_t = states[:, t].to(device)
        act_t = actions[:, t].to(device)
        embed = wm.encoder(img_t, st_t)
        post, _ = wm.rssm.obs_step(prev, act_t, embed)
        posts.append(post)
        prev = post
    return posts


@torch.no_grad()
def _rollout_from_start(online_wm, start_state, actions, context_idx: int, max_h: int, device):
    state_roll = start_state
    by_horizon = {}
    for h in range(1, max_h + 1):
        act_h = actions[:, context_idx + h].to(device)
        state_roll = online_wm.rssm.img_step(state_roll, act_h)
        by_horizon[h] = state_roll
    return by_horizon


@torch.no_grad()
def _fixed_steer_rollout(
    online_wm,
    start_state,
    actions,
    context_idx: int,
    max_h: int,
    steer: float,
    device,
):
    state_roll = start_state
    base_action = actions[:, context_idx].to(device).clone()
    if base_action.shape[-1] > 0:
        base_action[:, 0] = steer
    by_horizon = {}
    for h in range(1, max_h + 1):
        state_roll = online_wm.rssm.img_step(state_roll, base_action)
        by_horizon[h] = state_roll
    return by_horizon


@torch.no_grad()
def _collect_visuals(agent: DeepAIFAgent, images, states, actions, context_idx, horizons):
    online_wm = agent.world_model
    target_wm = agent.target_world_model
    if target_wm is None:
        raise RuntimeError("target_world_model is not loaded; pass --init_from.")
    device = agent._device

    images = images.to(device)
    states = states.to(device)
    actions = actions.to(device)
    max_h = max(horizons)
    bsz = images.shape[0]

    target_posts = _posterior_sequence(target_wm, images, states, actions, device)
    online_posts = _posterior_sequence(online_wm, images, states, actions, device)

    target_roll = _rollout_from_start(
        online_wm, target_posts[context_idx], actions, context_idx, max_h, device,
    )
    online_roll = _rollout_from_start(
        online_wm, online_posts[context_idx], actions, context_idx, max_h, device,
    )
    steer_plus = _fixed_steer_rollout(
        online_wm, target_posts[context_idx], actions, context_idx, max_h, 0.2, device,
    )
    steer_minus = _fixed_steer_rollout(
        online_wm, target_posts[context_idx], actions, context_idx, max_h, -0.2, device,
    )

    rows = {
        "GT": [],
        "target_wm posterior recon": [],
        "online_wm posterior recon": [],
        "target-start rollout": [],
        "online-start rollout": [],
        "steer +0.2 rollout from target start": [],
        "steer -0.2 rollout from target start": [],
    }
    gt_metric = []
    target_recon_metric = []
    online_recon_metric = []
    target_roll_metric = []
    online_roll_metric = []
    plus_metric = []
    minus_metric = []

    for b in range(bsz):
        for h in horizons:
            idx = context_idx + h
            gt = online_wm.preprocess_image(images[b:b + 1, idx]).squeeze(0)
            target_recon = target_wm.decode_obs(
                _det_state(_slice_state(target_posts[idx], b))
            ).squeeze(0)
            online_recon = online_wm.decode_obs(
                _det_state(_slice_state(online_posts[idx], b))
            ).squeeze(0)
            target_img = online_wm.decode_obs(
                _det_state(_slice_state(target_roll[h], b))
            ).squeeze(0)
            online_img = online_wm.decode_obs(
                _det_state(_slice_state(online_roll[h], b))
            ).squeeze(0)
            plus_img = online_wm.decode_obs(
                _det_state(_slice_state(steer_plus[h], b))
            ).squeeze(0)
            minus_img = online_wm.decode_obs(
                _det_state(_slice_state(steer_minus[h], b))
            ).squeeze(0)

            rows["GT"].append(gt)
            rows["target_wm posterior recon"].append(target_recon)
            rows["online_wm posterior recon"].append(online_recon)
            rows["target-start rollout"].append(target_img)
            rows["online-start rollout"].append(online_img)
            rows["steer +0.2 rollout from target start"].append(plus_img)
            rows["steer -0.2 rollout from target start"].append(minus_img)

            gt_metric.append(gt)
            target_recon_metric.append(target_recon)
            online_recon_metric.append(online_recon)
            target_roll_metric.append(target_img)
            online_roll_metric.append(online_img)
            plus_metric.append(plus_img)
            minus_metric.append(minus_img)

    def stack(xs):
        return torch.stack([x.detach().float() for x in xs], dim=0)

    gt_s = stack(gt_metric)
    target_recon_s = stack(target_recon_metric)
    online_recon_s = stack(online_recon_metric)
    target_roll_s = stack(target_roll_metric)
    online_roll_s = stack(online_roll_metric)
    plus_s = stack(plus_metric)
    minus_s = stack(minus_metric)

    metrics = {
        "target_recon_mse": (target_recon_s - gt_s).pow(2).mean(),
        "online_recon_mse": (online_recon_s - gt_s).pow(2).mean(),
        "target_start_rollout_mse": (target_roll_s - gt_s).pow(2).mean(),
        "online_start_rollout_mse": (online_roll_s - gt_s).pow(2).mean(),
        "target_start_rollout_var": target_roll_s.var(dim=0, unbiased=False).mean(),
        "online_start_rollout_var": online_roll_s.var(dim=0, unbiased=False).mean(),
        "gt_var": gt_s.var(dim=0, unbiased=False).mean(),
        "action_sensitivity_left_right_mse": (plus_s - minus_s).pow(2).mean(),
    }
    return rows, metrics


def _save_grid(rows: dict[str, list[torch.Tensor]], horizons: list[int], num_samples: int, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_names = list(rows.keys())
    n_rows = len(row_names)
    n_cols = num_samples * len(horizons)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.2 * n_cols, 6.0), 2.0 * n_rows),
        dpi=130,
        squeeze=False,
    )

    for r, name in enumerate(row_names):
        imgs = rows[name]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            ax.imshow(_to_display(imgs[c]), interpolation="nearest")
            if r == 0:
                sample_idx = c // len(horizons) + 1
                horizon = horizons[c % len(horizons)]
                ax.set_title(f"S{sample_idx} t+{horizon}", fontsize=9)
            if c == 0:
                ax.text(
                    -0.03,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--valid_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--context_frames", type=int, default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--diag_obstacle_visible", action="store_true")
    parser.add_argument("--stage", default="transition")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    cfg.training.stage = args.stage
    context_frames = (
        int(args.context_frames)
        if args.context_frames is not None
        else int(getattr(cfg.training, "rollout_context_frames", 1))
    )
    context_frames = max(1, min(context_frames, int(cfg.training.seq_len)))
    context_idx = context_frames - 1

    rollout_horizon = int(getattr(cfg.training, "rollout_horizon", 1))
    requested_horizons = _parse_horizons(args.horizons) or _default_horizons(cfg)
    horizons = [
        int(h)
        for h in requested_horizons
        if 1 <= int(h) <= rollout_horizon and context_idx + int(h) < int(cfg.training.seq_len)
    ]
    if not horizons:
        horizons = [1]

    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    if args.stage == "transition":
        agent.freeze_encoder_decoder()
    agent.set_target_world_model_from_checkpoint(args.init_from)
    agent.world_model.eval()
    if agent.target_world_model is not None:
        agent.target_world_model.eval()

    images, states, actions = _select_windows(
        args.valid_data,
        cfg,
        num_samples=args.num_samples,
        context_idx=context_idx,
        horizons=horizons,
        obstacle_visible=bool(args.diag_obstacle_visible),
    )

    rows, metrics = _collect_visuals(
        agent,
        images,
        states,
        actions,
        context_idx=context_idx,
        horizons=horizons,
    )

    output_path = Path(args.output_dir) / "target_rollout_visual.png"
    _save_grid(rows, horizons, images.shape[0], output_path)

    print(f"checkpoint: {args.checkpoint}")
    print(f"init_from: {args.init_from}")
    print(f"valid_data: {args.valid_data}")
    print(f"output: {output_path}")
    print(f"num_samples: {images.shape[0]}")
    print(f"context_frames: {context_frames}")
    print(f"context_idx: {context_idx}")
    print(f"horizons: {horizons}")
    print(f"diag_obstacle_visible: {args.diag_obstacle_visible}")
    for key in [
        "target_recon_mse",
        "online_recon_mse",
        "target_start_rollout_mse",
        "online_start_rollout_mse",
        "target_start_rollout_var",
        "online_start_rollout_var",
        "gt_var",
        "action_sensitivity_left_right_mse",
    ]:
        print(f"{key}: {float(metrics[key].detach().cpu()):.8f}")


if __name__ == "__main__":
    main()
