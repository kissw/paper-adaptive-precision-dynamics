#!/usr/bin/env python3
"""Apply action-indexing and validation-loss fixes.

Run from repository root:
    uv run python scripts/apply_action_validation_fix.py

This modifies:
    - src/active_inference/agent.py
    - scripts/train.py
"""

from __future__ import annotations

from pathlib import Path


REPO = Path.cwd()
AGENT = REPO / "src/active_inference/agent.py"
TRAIN = REPO / "scripts/train.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"[{label}] target block not found. File may have changed already.")
    return text.replace(old, new, 1)


def replace_all_required(text: str, old: str, new: str, label: str, expected_min: int = 1) -> str:
    count = text.count(old)
    if count < expected_min:
        raise RuntimeError(f"[{label}] expected at least {expected_min} occurrence(s), found {count}.")
    return text.replace(old, new)


def patch_agent() -> None:
    text = AGENT.read_text(encoding="utf-8")

    old = """        prev_state = wm.rssm.initial(B, self._device)
        prev_action = torch.zeros(
            B, self._cfg.cem.action_dim, device=self._device,
        )

        for t in range(T):
            img_raw_t = images[:, t].to(self._device)
            img_t = wm.preprocess_image(img_raw_t)  # encoder input == recon target
            st_t = states[:, t].to(self._device)
            act_t = (
                actions[:, t].to(self._device) if t > 0
                else prev_action
            )

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(
                    prev_state, prev_action, embed,
                )
"""
    new = """        # Dataset convention used by the v5 collectors:
        #   action[t] is the action applied by env.step(action[t]) that produced
        #   image[t], state[t]. Therefore obs_step at timestep t must be
        #   conditioned on actions[:, t], not on a one-step-delayed prev_action.
        prev_state = wm.rssm.initial(B, self._device)

        for t in range(T):
            img_raw_t = images[:, t].to(self._device)
            img_t = wm.preprocess_image(img_raw_t)  # encoder input == recon target
            st_t = states[:, t].to(self._device)
            act_t = actions[:, t].to(self._device)

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(
                    prev_state, act_t, embed,
                )
"""
    text = replace_once(text, old, new, "agent.update action indexing")

    old_prev_action_update = """                prev_action = (
                    act_t.detach() if t > 0 else prev_action
                )
"""
    text = replace_all_required(
        text,
        old_prev_action_update,
        "",
        "remove delayed prev_action updates in update()",
        expected_min=1,
    )

    old = """            state = wm.rssm.initial(B, self._device)
            prev_act = torch.zeros(B, self._cfg.cem.action_dim, device=self._device)

            for t in range(T):
                embed = wm.encode_obs(
                    images[:, t].to(self._device),
                    states[:, t].to(self._device),
                )
                post, _ = wm.rssm.obs_step(state, prev_act, embed)
                latents.append(post.mean.cpu())
                state = type(post)(*[x.detach() for x in post])
                prev_act = actions[:, t].to(self._device)
"""
    new = """            state = wm.rssm.initial(B, self._device)

            for t in range(T):
                act_t = actions[:, t].to(self._device)
                embed = wm.encode_obs(
                    images[:, t].to(self._device),
                    states[:, t].to(self._device),
                )
                post, _ = wm.rssm.obs_step(state, act_t, embed)
                latents.append(post.mean.cpu())
                state = type(post)(*[x.detach() for x in post])
"""
    text = replace_once(text, old, new, "encode_preference_data action indexing")

    AGENT.write_text(text, encoding="utf-8")
    print(f"patched: {AGENT}")


EVALUATE_FUNCTION = r"""

@torch.no_grad()
def evaluate_world_model(agent: DeepAIFAgent, dataloader) -> dict[str, float]:
    """ + '"""' + r"""Evaluate world-model VFE on a held-out HDF5 split.

    This mirrors DeepAIFAgent.update() but does not run optimizer/backward.

    Dataset convention:
        action[t] is the action applied by env.step(action[t]) that produced
        image[t], state[t]. Therefore obs_step at timestep t is conditioned on
        actions[:, t], matching the fixed training path.
    """ + '"""' + r"""
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
"""


def patch_train() -> None:
    text = TRAIN.read_text(encoding="utf-8")

    if "from active_inference.training.losses import compute_vfe" not in text:
        text = replace_once(
            text,
            "from active_inference.utils.seed import set_seed\n",
            "from active_inference.utils.seed import set_seed\n"
            "from active_inference.training.losses import compute_vfe\n",
            "train import compute_vfe",
        )

    text = replace_once(
        text,
        """    train_loss: float | None,
    best_epoch: int | None,
""",
        """    train_loss: float | None,
    val_loss: float | None,
    best_epoch: int | None,
""",
        "build_checkpoint signature val_loss",
    )

    text = replace_once(
        text,
        """        "train_loss":       float(train_loss) if train_loss is not None else None,
""",
        """        "train_loss":       float(train_loss) if train_loss is not None else None,
        "val_loss":         float(val_loss)   if val_loss   is not None else None,
""",
        "checkpoint val_loss metadata",
    )

    if "def evaluate_world_model(" not in text:
        text = replace_once(
            text,
            "\n\ndef main():\n",
            EVALUATE_FUNCTION + "\n\ndef main():\n",
            "insert evaluate_world_model",
        )

    if 'parser.add_argument("--valid_data"' not in text:
        text = replace_once(
            text,
            """    parser.add_argument("--output_dir", default="outputs/train")
""",
            """    parser.add_argument("--output_dir", default="outputs/train")
    parser.add_argument("--valid_data", default=None, help="Optional held-out HDF5 validation split")
""",
            "add --valid_data argument",
        )

    old = """    dataloader = get_dataloader(
        data_path,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=0,
        shuffle=True,
    )

    agent = DeepAIFAgent(cfg)
"""
    new = """    dataloader = get_dataloader(
        data_path,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=0,
        shuffle=True,
    )

    valid_loader = None
    if args.valid_data is not None:
        valid_loader = get_dataloader(
            args.valid_data,
            batch_size=cfg.training.batch_size,
            seq_len=cfg.training.seq_len,
            num_workers=0,
            shuffle=False,
        )
        print(f"Using validation data: {args.valid_data}")
        print(f"Validation sequences: {len(valid_loader.dataset)}")

    agent = DeepAIFAgent(cfg)
"""
    text = replace_once(text, old, new, "insert valid_loader")

    text = replace_once(
        text,
        """    mean_loss: float = 0.0

    for epoch in range(start_epoch, cfg.training.epochs):
""",
        """    mean_loss: float = 0.0
    last_val_loss: float | None = None

    for epoch in range(start_epoch, cfg.training.epochs):
""",
        "last_val_loss init",
    )

    old = """        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch + 1
            torch.save(
                build_checkpoint(
                    agent, cfg, args,
                    epoch=epoch + 1,
                    global_step=global_step,
                    train_loss=mean_loss,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    checkpoint_type="best",
                    is_best=True,
                ),
                str(ckpt_dir / "best.pt"),
            )
"""
    new = """        val_info = None
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

        if selection_loss < best_loss:
            best_loss = selection_loss
            best_epoch = epoch + 1
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
"""
    text = replace_once(text, old, new, "replace best checkpoint block")

    text = replace_all_required(
        text,
        """                train_loss=mean_loss,
                best_epoch=best_epoch,
""",
        """                train_loss=mean_loss,
                val_loss=last_val_loss,
                best_epoch=best_epoch,
""",
        "epoch checkpoint val_loss",
        expected_min=1,
    )
    text = replace_all_required(
        text,
        """            train_loss=mean_loss if epoch_losses else None,
            best_epoch=best_epoch,
""",
        """            train_loss=mean_loss if epoch_losses else None,
            val_loss=last_val_loss,
            best_epoch=best_epoch,
""",
        "final checkpoint val_loss",
        expected_min=1,
    )

    TRAIN.write_text(text, encoding="utf-8")
    print(f"patched: {TRAIN}")


def main() -> None:
    if not AGENT.exists():
        raise FileNotFoundError(AGENT)
    if not TRAIN.exists():
        raise FileNotFoundError(TRAIN)

    patch_agent()
    patch_train()
    print("Done. Now run:")
    print("  uv run python -m py_compile src/active_inference/agent.py scripts/train.py")


if __name__ == "__main__":
    main()