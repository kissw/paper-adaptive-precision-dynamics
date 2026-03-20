import argparse
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.data.dataset import get_dataloader, get_preference_dataloader
from active_inference.data.synthetic import SyntheticDrivingData
from active_inference.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="outputs/train")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch number")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.device:
        cfg.device = args.device

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
        num_workers=0,
        shuffle=True,
    )

    agent = DeepAIFAgent(cfg)

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

    global_step = start_epoch * len(dataloader)
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_losses = []

        for batch_idx, (images, states, actions) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", leave=False)
        ):
            try:
                info = agent.update(images, states, actions)
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

        if mean_loss < best_loss:
            best_loss = mean_loss
            agent.save_checkpoint(str(ckpt_dir / "best.pt"))
        agent.save_checkpoint(str(ckpt_dir / f"epoch_{epoch + 1}.pt"))

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

    agent.save_checkpoint(str(ckpt_dir / "final.pt"))
    writer.close()
    print(f"Training complete. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
