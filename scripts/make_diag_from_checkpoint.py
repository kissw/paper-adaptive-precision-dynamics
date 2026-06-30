import argparse
from pathlib import Path

import torch

from active_inference.agent import DeepAIFAgent
from active_inference.config import Config
from active_inference.training.diag_grid import make_diag_grid


def _select_diag_window(h5_path: str, seq_len: int, start_idx: int | None):
    """Pick a fixed same-episode window for diagnostic rollout rendering."""
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        episode_ids = f["episode_ids"][:]
        obs_vis = (
            f["obstacle_visible"][:].astype(bool)
            if "obstacle_visible" in f else None
        )
        n = len(episode_ids)

        if start_idx is None and obs_vis is not None and obs_vis.any():
            visible = np.where(obs_vis)[0]
            anchor = int(visible[len(visible) // 2])
        elif start_idx is not None:
            anchor = int(start_idx)
        else:
            anchor = n // 2

        ep = episode_ids[anchor]
        ep_idxs = np.where(episode_ids == ep)[0]
        ep_start, ep_end = int(ep_idxs[0]), int(ep_idxs[-1])
        win_start = max(ep_start, min(anchor - seq_len // 3, ep_end - seq_len + 1))
        win_start = max(ep_start, win_start)
        win_end = min(win_start + seq_len, ep_end + 1)
        if win_end - win_start < 2:
            return None

        images = torch.from_numpy(f["images"][win_start:win_end]).float()
        states = torch.from_numpy(f["states"][win_start:win_end]).float()
        actions = torch.from_numpy(f["actions"][win_start:win_end]).float()

    local_start = max(0, anchor - win_start)
    local_start = min(local_start, (win_end - win_start) - 2)
    return images, states, actions, local_start


def _parse_horizons(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    horizons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return horizons or None


def _diagnostic_settings(args, cfg: Config) -> tuple[int, list[int]]:
    is_target_rollout = (
        args.stage == "transition"
        and getattr(cfg.training, "transition_loss_mode", "") == "target_rollout"
    )

    cli_horizons = _parse_horizons(args.diag_horizons)
    if not is_target_rollout:
        return (
            min(5, int(cfg.training.seq_len)),
            cli_horizons if cli_horizons is not None else [1, 5, 10, 15],
        )

    rollout_horizon = int(getattr(cfg.training, "rollout_horizon", 1))
    diag_context_len = min(
        int(getattr(cfg.training, "rollout_context_frames", cfg.training.seq_len)),
        int(cfg.training.seq_len),
    )

    if cli_horizons is not None:
        horizons = cli_horizons
    else:
        configured_horizons = getattr(cfg.training, "rollout_decode_horizons", None)
        if configured_horizons:
            horizons = [int(h) for h in configured_horizons]
        else:
            horizons = [1, rollout_horizon]

    horizons = [h for h in horizons if 1 <= h <= rollout_horizon]
    if not horizons:
        horizons = [1]
    return diag_context_len, horizons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--valid_data", default=None, help="Held-out HDF5 data path")
    parser.add_argument("--data", default=None, help="Training HDF5 data path")
    parser.add_argument("--output", required=True, help="Output diagnostic image path")
    parser.add_argument("--stage", default="transition", choices=["joint", "ae", "transition"])
    parser.add_argument("--diag_start_idx", type=int, default=None)
    parser.add_argument("--diag_horizons", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides applied after loading --config.",
    )
    args = parser.parse_args()

    data_path = args.valid_data or args.data
    if data_path is None:
        parser.error("Provide --valid_data or --data for the diagnostic window.")

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    cfg.training.stage = args.stage
    if args.device:
        cfg.device = args.device

    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    if args.stage == "transition":
        agent.freeze_encoder_decoder()
    agent.world_model.eval()

    diag_window = _select_diag_window(
        data_path,
        int(cfg.training.seq_len),
        args.diag_start_idx,
    )
    if diag_window is None:
        raise RuntimeError("Could not select a diagnostic window with enough frames.")

    diag_context_len, diag_horizons = _diagnostic_settings(args, cfg)
    d_imgs, d_states, d_actions, d_start = diag_window
    ckpt_name = Path(args.checkpoint).name
    title = (
        f"{ckpt_name}  context={diag_context_len}  "
        f"horizons={','.join(str(h) for h in diag_horizons)}"
    )

    ok = make_diag_grid(
        agent.world_model,
        d_imgs,
        d_states,
        d_actions,
        start_idx=d_start,
        output_path=args.output,
        horizons=diag_horizons,
        context_len=diag_context_len,
        device=agent._device,
        title=title,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {data_path}")
    print(f"output: {args.output}")
    print(f"diag_context_len: {diag_context_len}")
    print(f"diag_horizons: {diag_horizons}")
    print(f"local_start: {d_start}")
    print(f"saved: {bool(ok)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
