import argparse
import dataclasses
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.data.dataset import get_dataloader, get_preference_dataloader
from active_inference.data.synthetic import SyntheticDrivingData
from active_inference.training.losses import compute_vfe
from active_inference.training.diag_grid import make_diag_grid
from active_inference.utils.seed import set_seed


def _select_diag_window(h5_path: str, seq_len: int, start_idx: int | None):
    """Pick a fixed (images, states, actions, window_start_idx) for diagnostics.

    Chooses a contiguous same-episode window of length seq_len.  When start_idx
    is None, anchors on the first frame whose obstacle_visible flag is True so
    the obstacle is in view; the rollout begins partway into the window.
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        episode_ids = f["episode_ids"][:]
        obs_vis = (
            f["obstacle_visible"][:].astype(bool)
            if "obstacle_visible" in f else None
        )
        n = len(episode_ids)

        # Determine the anchor frame
        if start_idx is None and obs_vis is not None and obs_vis.any():
            anchor = int(np.where(obs_vis)[0][len(np.where(obs_vis)[0]) // 2])
        elif start_idx is not None:
            anchor = int(start_idx)
        else:
            anchor = n // 2

        # Build a same-episode window of length seq_len ending well past anchor
        ep = episode_ids[anchor]
        ep_idxs = np.where(episode_ids == ep)[0]
        ep_start, ep_end = int(ep_idxs[0]), int(ep_idxs[-1])
        win_start = max(ep_start, min(anchor - seq_len // 3, ep_end - seq_len + 1))
        win_start = max(ep_start, win_start)
        win_end = min(win_start + seq_len, ep_end + 1)
        if win_end - win_start < 2:
            return None

        images  = torch.from_numpy(f["images"][win_start:win_end]).float()
        states  = torch.from_numpy(f["states"][win_start:win_end]).float()
        actions = torch.from_numpy(f["actions"][win_start:win_end]).float()

    # Rollout start position within the window
    local_start = max(0, anchor - win_start)
    local_start = min(local_start, (win_end - win_start) - 2)
    return images, states, actions, local_start


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _safe_config_to_dict(cfg) -> dict:
    """Recursively convert Config dataclass to a plain serialisable dict."""
    try:
        return dataclasses.asdict(cfg)
    except Exception:
        return {}


def _get_git_info() -> tuple[str | None, str | None]:
    """Return (commit_hash, branch_name) from git, or (None, None) on failure."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return commit, branch
    except Exception:
        return None, None


def build_checkpoint(
    agent: DeepAIFAgent,
    cfg: Config,
    args,
    epoch: int,
    global_step: int | None,
    train_loss: float | None,
    val_loss: float | None,
    best_epoch: int | None,
    best_loss: float | None,
    checkpoint_type: str,
    is_best: bool,
) -> dict:
    """Build a checkpoint dict with metadata.

    Preserves the existing top-level keys (world_model, optimizer, preference)
    so that all downstream scripts continue to work unchanged.
    """
    git_commit, git_branch = _get_git_info()

    pref = agent.preference
    try:
        pref_dict = {
            "means": pref.means.data,
            "log_stds": pref.log_stds.data,
            "logits": pref.logits.data,
        }
    except AttributeError:
        pref_dict = {}

    return {
        "world_model": agent.world_model.state_dict(),
        "optimizer": agent._optimizer.state_dict(),
        "preference": pref_dict,
        "epoch": int(epoch),
        "global_step": int(global_step) if global_step is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_metric": float(best_loss) if best_loss is not None else None,
        "best_loss": float(best_loss) if best_loss is not None else None,
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "config": _safe_config_to_dict(cfg),
        "world_model_type": getattr(getattr(cfg, "model", None), "world_model_type", None),
        "crop_road": getattr(getattr(cfg, "encoder", None), "crop_road", None),
        "image_size": getattr(getattr(cfg, "encoder", None), "image_size", None),
        "data_path": getattr(args, "data", None),
        "valid_data_path": getattr(args, "valid_data", None),
        "output_dir": getattr(args, "output_dir", None),
        "save_interval": getattr(args, "save_interval", None),
        "early_stop_patience": getattr(args, "early_stop_patience", None),
        "early_stop_min_delta": getattr(args, "early_stop_min_delta", None),
        "checkpoint_type": checkpoint_type,
        "is_best": bool(is_best),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_branch": git_branch,
    }


@torch.no_grad()
def evaluate_world_model(agent: DeepAIFAgent, dataloader) -> dict[str, float]:
    """Evaluate world-model VFE on a held-out HDF5 split.

    This mirrors DeepAIFAgent.update() without optimizer/backward.

    Dataset convention:
        action[t] is the action applied by env.step(action[t]) that produced
        image[t], state[t]. Therefore obs_step at timestep t is conditioned on
        actions[:, t], matching the fixed training path in agent.update().
    """
    wm = agent.world_model
    device = agent._device
    cfg = agent._cfg.training
    beta_obstacle_aux = getattr(cfg, "beta_obstacle_aux", 0.0)

    was_training = wm.training
    wm.eval()

    n_batches = 0
    totals = {
        "total_loss": 0.0,
        "img_loss": 0.0,
        "state_loss": 0.0,
        "kl_dyn": 0.0,
        "kl_rep": 0.0,
        "obs_aux_loss": 0.0,
    }

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        if len(batch) == 4:
            images, states, actions, obs_labels = batch
        else:
            images, states, actions = batch
            obs_labels = None

        B, T = images.shape[0], images.shape[1]
        prev_state = wm.rssm.initial(B, device)

        batch_total = 0.0
        batch_accum = {
            "img_loss": 0.0,
            "state_loss": 0.0,
            "kl_dyn": 0.0,
            "kl_rep": 0.0,
            "obs_aux_loss": 0.0,
        }
        valid_steps = 0

        for t in range(T):
            img_raw_t = images[:, t].to(device)
            img_t = wm.preprocess_image(img_raw_t)
            st_t = states[:, t].to(device)
            act_t = actions[:, t].to(device)

            with torch.amp.autocast("cuda", enabled=agent._use_amp):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(prev_state, act_t, embed)

                feat = wm.rssm.get_feat(post)
                recon_img = wm.decode_obs(post)
                recon_state = wm.state_decoder(feat)

                post_mean, post_std = wm.get_kl_stats(post)
                prior_mean, prior_std = wm.get_kl_stats(prior)
                loss, info = compute_vfe(
                    post_mean,
                    post_std,
                    prior_mean,
                    prior_std,
                    img_t,
                    recon_img,
                    st_t,
                    recon_state,
                    free_nats=cfg.free_nats,
                    kl_dyn_scale=cfg.kl_dyn_scale,
                    kl_rep_scale=cfg.kl_rep_scale,
                )

                obs_aux_loss_val = 0.0
                if obs_labels is not None and beta_obstacle_aux > 0:
                    obs_logit = wm.obstacle_head(feat).squeeze(-1)
                    obs_label = obs_labels[:, t].to(device).float()
                    obs_aux = torch.nn.functional.binary_cross_entropy_with_logits(
                        obs_logit,
                        obs_label,
                    )
                    loss = loss + beta_obstacle_aux * obs_aux
                    obs_aux_loss_val = obs_aux.item()

            if torch.isnan(loss) or torch.isinf(loss):
                prev_state = type(post)(*[x.detach() for x in post])
                continue

            batch_total += loss.item()
            for k in ["img_loss", "state_loss", "kl_dyn", "kl_rep"]:
                batch_accum[k] += info[k].item()
            batch_accum["obs_aux_loss"] += obs_aux_loss_val
            valid_steps += 1

            prev_state = type(post)(*[x.detach() for x in post])

        if valid_steps > 0:
            n_batches += 1
            totals["total_loss"] += batch_total / valid_steps
            for k in batch_accum:
                totals[k] += batch_accum[k] / valid_steps

    if was_training:
        wm.train()

    if n_batches == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--valid_data", default=None, help="Optional held-out HDF5 validation split")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="outputs/train")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch number")

    parser.add_argument(
        "--save_interval",
        "--save-interval",
        type=int,
        default=1,
        help=(
            "Save numbered epoch checkpoints every N epochs. "
            "1 saves every epoch, 10 saves epoch_10.pt/epoch_20.pt/... . "
            "0 disables periodic epoch checkpoint saving. best.pt and final.pt are always saved."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "Early-stop patience in epochs. "
            "0 disables early stopping. Uses validation loss when --valid_data is provided, otherwise train loss."
        ),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum loss decrease required to reset early-stop patience.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "OmegaConf dotlist overrides applied after loading --config. "
            "Example: --set training.overshoot_horizon=5 training.overshoot_weight=0.5"
        ),
    )
    parser.add_argument(
        "--diag_grid", action="store_true", default=False,
        help="Save a 5-row rollout diagnostic grid each time best.pt improves.",
    )
    parser.add_argument(
        "--diag_start_idx", type=int, default=None,
        help="Frame index (within validation data) to anchor the diagnostic grid. "
             "Default: auto-pick a frame with a visible obstacle.",
    )
    parser.add_argument(
        "--diag_horizons", default="1,5,10,15",
        help="Comma-separated horizons for the diagnostic grid columns.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10,
        help="DataLoader CPU worker 개수 (also caps torch intra-op threads)",
    )
    parser.add_argument(
        "--stage", choices=["joint", "ae", "transition"], default="joint",
        help="Two-stage training: joint (default, original) | "
             "ae (encoder-decoder only) | transition (rssm only, enc-dec frozen)",
    )
    parser.add_argument(
        "--init_from", default=None,
        help="Stage-1 (ae) checkpoint to initialise encoder-decoder weights "
             "from, used in --stage transition.",
    )
    parser.add_argument(
        "--ae_kl_rep", type=float, default=0.01,
        help="Stage-ae weak posterior KL toward N(0,1); 0 = pure AE.",
    )

    args = parser.parse_args()

    if args.save_interval < 0:
        raise ValueError("--save_interval must be >= 0")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be >= 0")

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.device:
        cfg.device = args.device
    # Two-stage training settings (must be set before agent construction so the
    # stage-aware optimizer / freeze logic in DeepAIFAgent.__init__ applies).
    cfg.training.stage = args.stage
    cfg.training.ae_kl_rep = args.ae_kl_rep

    # Cap CPU thread usage to avoid over-subscription on shared machines.
    if args.num_workers > 0:
        torch.set_num_threads(args.num_workers)

    set_seed(cfg.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if args.data is None:
        print("No data provided, generating synthetic data...")
        synth = SyntheticDrivingData()
        data = synth.generate(n_episodes=10, episode_len=cfg.training.seq_len * 2)
        data_path = str(output_dir / "synthetic.h5")
        synth.to_hdf5(data, data_path)
    else:
        data_path = args.data

    dataloader = get_dataloader(
        data_path,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=args.num_workers,
        shuffle=True,
    )

    valid_loader = None
    if args.valid_data is not None:
        valid_loader = get_dataloader(
            args.valid_data,
            batch_size=cfg.training.batch_size,
            seq_len=cfg.training.seq_len,
            num_workers=args.num_workers,
            shuffle=False,
        )
        print(f"Using validation data: {args.valid_data}")
        print(f"Validation sequences: {len(valid_loader.dataset)}")

    agent = DeepAIFAgent(cfg)
    print(f"Training stage: {args.stage}")

    # Stage-2 (transition): initialise encoder-decoder from a stage-1 ckpt.
    if args.stage == "transition":
        if args.init_from is None:
            print("WARNING: --stage transition without --init_from; "
                  "encoder-decoder will use random init (not recommended).")
        else:
            agent.load_encoder_decoder(args.init_from)
            agent.freeze_encoder_decoder()  # re-assert eval()/requires_grad after load

    start_epoch = args.start_epoch
    if args.resume:
        print(f"Resuming from {args.resume}")
        agent.load_checkpoint(args.resume)
        if start_epoch == 0:
            import re

            m = re.search(r"epoch_(\d+)", args.resume)
            if m:
                start_epoch = int(m.group(1))
                print(f"Auto-detected start epoch: {start_epoch}")

    writer = SummaryWriter(str(output_dir / "tb_logs"))

    # Fixed diagnostic window (selected once so successive grids are comparable)
    diag_window = None
    diag_dir = output_dir / "diag"
    diag_horizons = [int(h) for h in args.diag_horizons.split(",")]
    if args.diag_grid:
        diag_source = args.valid_data if args.valid_data is not None else data_path
        try:
            diag_window = _select_diag_window(
                diag_source, cfg.training.seq_len, args.diag_start_idx,
            )
            if diag_window is not None:
                print(f"Diagnostic grid enabled: source={diag_source} "
                      f"window_len={diag_window[0].shape[0]} "
                      f"local_start={diag_window[3]}")
        except Exception as e:
            print(f"WARNING: diag window selection failed ({e}); disabling diag grid")
            diag_window = None

    global_step = start_epoch * len(dataloader)
    best_loss = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False

    epoch_losses: list[float] = []
    mean_loss: float = 0.0
    last_val_loss: float | None = None
    completed_epoch = start_epoch

    print("=" * 100)
    print("Training configuration")
    print(f"data: {data_path}")
    print(f"valid_data: {args.valid_data}")
    print(f"epochs: {cfg.training.epochs}")
    print(f"batch_size: {cfg.training.batch_size}")
    print(f"seq_len: {cfg.training.seq_len}")
    print(f"batches_per_epoch: {len(dataloader)}")
    print(f"save_interval: {args.save_interval}")
    print(f"early_stop_patience: {args.early_stop_patience}")
    print(f"early_stop_min_delta: {args.early_stop_min_delta}")
    print("=" * 100)

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_losses = []
        completed_epoch = epoch + 1

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", leave=False)
        ):
            # Dataset returns 3 or 4 tensors depending on obstacle labels
            if len(batch) == 4:
                images, states, actions, obs_labels = batch
            else:
                images, states, actions = batch
                obs_labels = None
            try:
                if args.stage == "ae":
                    info = agent.update_ae(images, states, actions, obs_labels)
                else:
                    info = agent.update(images, states, actions, obs_labels)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"\nCUDA error at epoch {epoch + 1}, batch {batch_idx}: {e}")
                    print("Saving checkpoint and skipping batch...")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    agent.save_checkpoint(str(ckpt_dir / f"crash_epoch_{epoch + 1}.pt"))
                    continue
                raise

            if torch.isnan(torch.tensor(info["total_loss"])):
                print(f"NaN at epoch {epoch + 1}, batch {batch_idx}. Skipping batch.")
                continue

            epoch_losses.append(info["total_loss"])
            global_step += 1

            for k, v in info.items():
                writer.add_scalar(f"train/{k}", v, global_step)

        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | Loss: {mean_loss:.4f}")
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)

        if valid_loader is not None:
            val_info = evaluate_world_model(agent, valid_loader)
            last_val_loss = val_info["total_loss"]
            for k, v in val_info.items():
                writer.add_scalar(f"valid/{k}", v, epoch)
            print(
                f"  Validation | Loss: {last_val_loss:.4f} "
                f"img={val_info['img_loss']:.4f} "
                f"state={val_info['state_loss']:.4f} "
                f"kl_dyn={val_info['kl_dyn']:.4f} "
                f"kl_rep={val_info['kl_rep']:.4f}"
            )

        selection_loss = last_val_loss if valid_loader is not None else mean_loss

        improved = False
        if selection_loss is not None:
            loss_tensor = torch.tensor(selection_loss)
            if not torch.isnan(loss_tensor) and not torch.isinf(loss_tensor):
                improved = selection_loss < (best_loss - args.early_stop_min_delta)

        if improved:
            best_loss = float(selection_loss)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                build_checkpoint(
                    agent, cfg, args,
                    epoch=epoch + 1,
                    global_step=global_step,
                    train_loss=mean_loss,
                    val_loss=last_val_loss,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    checkpoint_type="best",
                    is_best=True,
                ),
                str(ckpt_dir / "best.pt"),
            )
            print(f"  New best checkpoint saved: epoch={best_epoch}, loss={best_loss:.6f}")

            # Diagnostic rollout grid (best.pt improved this epoch)
            if diag_window is not None:
                d_imgs, d_states, d_actions, d_start = diag_window
                try:
                    ok = make_diag_grid(
                        agent.world_model,
                        d_imgs, d_states, d_actions,
                        start_idx=d_start,
                        output_path=str(diag_dir / f"diag_epoch{best_epoch}_best.png"),
                        horizons=diag_horizons,
                        context_len=min(5, cfg.training.seq_len),
                        device=agent._device,
                        title=f"epoch {best_epoch}  val={last_val_loss}"
                              if last_val_loss is not None else f"epoch {best_epoch}",
                    )
                    if ok:
                        print(f"  Diagnostic grid → "
                              f"{diag_dir / f'diag_epoch{best_epoch}_best.png'}")
                except Exception as e:
                    print(f"  WARNING: diag grid failed: {e}")
        else:
            epochs_without_improvement += 1
            if best_loss < float("inf"):
                print(
                    f"  No improvement for {epochs_without_improvement} epoch(s) "
                    f"(best epoch={best_epoch}, best loss={best_loss:.6f})"
                )
            else:
                print(f"  No valid best loss yet. No-improvement count={epochs_without_improvement}")

        early_stop_triggered = (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
            and best_loss < float("inf")
        )

        should_save_epoch = (
            args.save_interval > 0
            and ((epoch + 1) % args.save_interval == 0)
        ) or early_stop_triggered

        if should_save_epoch:
            torch.save(
                build_checkpoint(
                    agent, cfg, args,
                    epoch=epoch + 1,
                    global_step=global_step,
                    train_loss=mean_loss,
                    val_loss=last_val_loss,
                    best_epoch=best_epoch,
                    best_loss=best_loss if best_loss < float("inf") else None,
                    checkpoint_type="epoch",
                    is_best=improved,
                ),
                str(ckpt_dir / f"epoch_{epoch + 1}.pt"),
            )
            print(f"  Epoch checkpoint saved: {ckpt_dir / f'epoch_{epoch + 1}.pt'}")

        # --- GMM preference update ---
        pref_cfg = cfg.preference
        should_update = (
            epoch + 1 >= pref_cfg.warmup_epoch
            and (epoch + 1 - pref_cfg.warmup_epoch) % pref_cfg.update_interval == 0
        )
        if should_update:
            pref_data = pref_cfg.data_path or data_path
            pref_loader = get_preference_dataloader(
                pref_data,
                batch_size=8,
                seq_len=cfg.training.seq_len,
                task_filter=None,
                success_only=True,
                num_workers=0,
            )
            if len(pref_loader.dataset) > 0:
                pref_info = agent.update_preference(pref_loader, max_samples=pref_cfg.max_samples)
                for k, v in pref_info.items():
                    writer.add_scalar(f"preference/{k}", v, epoch)
                print(
                    f"  GMM updated: log_prob={pref_info['pref_log_prob']:.2f} "
                    f"min_dist={pref_info['pref_min_component_dist']:.2f} "
                    f"samples={pref_info['pref_n_samples']}"
                )
                if pref_info["pref_min_component_dist"] < 0.5:
                    print("  WARNING: GMM components may be collapsing")
            else:
                print("  No valid preference sequences found (need success_flags in HDF5)")

        if early_stop_triggered:
            stopped_early = True
            print("=" * 100)
            print(
                "Early stopping triggered: "
                f"patience={args.early_stop_patience}, "
                f"min_delta={args.early_stop_min_delta}, "
                f"best_epoch={best_epoch}, "
                f"best_loss={best_loss:.6f}, "
                f"stop_epoch={epoch + 1}"
            )
            print("=" * 100)
            break

    torch.save(
        build_checkpoint(
            agent, cfg, args,
            epoch=completed_epoch,
            global_step=global_step,
            train_loss=mean_loss if epoch_losses else None,
            val_loss=last_val_loss,
            best_epoch=best_epoch,
            best_loss=best_loss if best_loss < float("inf") else None,
            checkpoint_type="final_early_stop" if stopped_early else "final",
            is_best=False,
        ),
        str(ckpt_dir / "final.pt"),
    )
    writer.close()
    print(f"Training complete. Checkpoints in {ckpt_dir}")
    if stopped_early:
        print(f"Stopped early at epoch {completed_epoch}. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
