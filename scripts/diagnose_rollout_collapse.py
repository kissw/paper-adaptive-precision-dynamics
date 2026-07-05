import argparse
from pathlib import Path

import torch

from active_inference.agent import DeepAIFAgent
from active_inference.config import Config
from active_inference.data.dataset import get_dataloader
from active_inference.utils.bbox import bbox_xyxy_to_mask, bbox_xyxy_to_model_space


def _to_display(img: torch.Tensor) -> torch.Tensor:
    return (img.detach().float().cpu() + 0.5).clamp(0.0, 1.0).permute(1, 2, 0)


def _err_display(err: torch.Tensor) -> torch.Tensor:
    e = err.detach().float().cpu().mean(dim=0)
    e = e / (e.max().clamp_min(1e-8))
    return e


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


def _decode_horizons(cfg: Config) -> list[int]:
    horizon = int(getattr(cfg.training, "rollout_horizon", 1))
    decode = getattr(cfg.training, "rollout_decode_horizons", None)
    if decode:
        horizons = [int(h) for h in decode]
    else:
        horizons = [1, horizon]
    horizons = [h for h in horizons if 1 <= h <= horizon]
    return horizons or [1]


def _bbox_model(bbox, cfg: Config, image_h: int, image_w: int):
    return bbox_xyxy_to_model_space(
        bbox,
        image_h=image_h,
        image_w=image_w,
        crop_road=bool(getattr(cfg.encoder, "crop_road", False)),
        keep_bottom_frac=float(getattr(cfg.encoder, "keep_bottom_frac", 0.6)),
        coord_space=getattr(cfg.training, "bbox_coord_space", "raw"),
        min_area=float(getattr(cfg.training, "bbox_min_area", 1.0)),
    )


@torch.no_grad()
def _collect_images(agent, images, states, actions, bboxes, max_samples: int):
    wm = agent.world_model
    target_wm = agent.target_world_model or agent.world_model
    cfg = agent._cfg
    device = agent._device
    B, T = images.shape[:2]
    Bv = min(B, max_samples)
    context = int(getattr(cfg.training, "rollout_context_frames", 25)) - 1
    context = max(0, min(context, T - 2))
    horizons = [h for h in _decode_horizons(cfg) if context + h < T]
    if not horizons:
        horizons = [1]

    target_posts = []
    online_posts = []
    prev_t = target_wm.rssm.initial(B, device)
    prev_o = wm.rssm.initial(B, device)
    for t in range(T):
        img_t = target_wm.preprocess_image(images[:, t].to(device))
        st_t = states[:, t].to(device)
        act_t = actions[:, t].to(device)
        emb_t = target_wm.encoder(img_t, st_t)
        post_t, _ = target_wm.rssm.obs_step(prev_t, act_t, emb_t)
        post_t = _det_state(post_t)
        target_posts.append(post_t)
        prev_t = post_t

        img_o = wm.preprocess_image(images[:, t].to(device))
        emb_o = wm.encoder(img_o, st_t)
        post_o, _ = wm.rssm.obs_step(prev_o, act_t, emb_o)
        post_o = _det_state(post_o)
        online_posts.append(post_o)
        prev_o = post_o

    prior = _det_state(target_posts[context])
    prior_by_h = {}
    for h in range(1, max(horizons) + 1):
        prior = _det_state(wm.rssm.img_step(prior, actions[:, context + h].to(device)))
        if h in horizons:
            prior_by_h[h] = prior

    rows = []
    for b in range(Bv):
        sample_rows = {
            "GT": [],
            "target posterior recon": [],
            "online posterior recon": [],
            "prior rollout": [],
            "abs error prior vs GT": [],
            "bbox zoom GT": [],
            "bbox zoom target": [],
            "bbox zoom prior": [],
            "bbox zoom error": [],
        }
        for h in horizons:
            idx = context + h
            gt = wm.preprocess_image(images[b:b + 1, idx].to(device)).squeeze(0)
            target = wm.decode_obs(
                _slice_state(target_posts[idx], b)
            ).squeeze(0)
            online = wm.decode_obs(
                _slice_state(online_posts[idx], b)
            ).squeeze(0)
            prior_img = wm.decode_obs(
                _slice_state(prior_by_h[h], b)
            ).squeeze(0)
            err = (prior_img - gt).abs()

            sample_rows["GT"].append(gt)
            sample_rows["target posterior recon"].append(target)
            sample_rows["online posterior recon"].append(online)
            sample_rows["prior rollout"].append(prior_img)
            sample_rows["abs error prior vs GT"].append(err)

            zooms = _bbox_zooms(
                [gt, target, prior_img, err],
                bboxes[b, idx] if bboxes is not None else None,
                cfg,
            )
            for key, zoom in zip(
                ["bbox zoom GT", "bbox zoom target", "bbox zoom prior", "bbox zoom error"],
                zooms,
            ):
                sample_rows[key].append(zoom)
        rows.append((b, sample_rows))
    return rows, horizons


def _slice_state(state, b: int):
    return type(state)(*[x[b:b + 1] for x in state])


def _bbox_zooms(imgs, bbox, cfg):
    if bbox is None:
        return imgs
    h, w = imgs[0].shape[-2:]
    model_bbox = _bbox_model(torch.as_tensor(bbox), cfg, h, w)
    mask, valid = bbox_xyxy_to_mask(model_bbox, h, w, dilate=8, min_area=1.0)
    if not valid.any() or mask.sum() == 0:
        return imgs
    ys, xs = mask[0, 0].nonzero(as_tuple=True)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return [img[:, y1:y2, x1:x2] for img in imgs]


def _save_grid(rows, horizons, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = sum(len(sample_rows) + 1 for _, sample_rows in rows)
    n_cols = len(horizons)
    fig, axes = plt.subplots(
        n_rows,
        n_cols + 1,
        figsize=((n_cols + 1) * 1.8, n_rows * 1.35),
        dpi=120,
        squeeze=False,
    )
    r = 0
    for sample_idx, sample_rows in rows:
        axes[r, 0].axis("off")
        axes[r, 0].text(0.95, 0.5, f"sample {sample_idx}", ha="right", va="center")
        for c in range(n_cols):
            axes[r, c + 1].axis("off")
        r += 1
        for label, imgs in sample_rows.items():
            axes[r, 0].axis("off")
            axes[r, 0].text(0.95, 0.5, label, ha="right", va="center", fontsize=8)
            for c, img in enumerate(imgs):
                ax = axes[r, c + 1]
                ax.axis("off")
                if label.endswith("error") or label == "abs error prior vs GT":
                    ax.imshow(_err_display(img), cmap="magma", vmin=0, vmax=1)
                else:
                    ax.imshow(_to_display(img), interpolation="nearest")
                if r == 1:
                    ax.set_title(f"t+{horizons[c]}", fontsize=8)
            r += 1
    fig.tight_layout(pad=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", default="transition", choices=["joint", "ae", "transition"])
    parser.add_argument("--init_from", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    cfg.training.stage = args.stage
    cfg.training.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device

    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    if args.stage == "transition":
        agent.freeze_encoder_decoder()
    if args.init_from:
        agent.set_target_world_model_from_checkpoint(args.init_from)
    elif getattr(cfg.training, "transition_loss_mode", "") == "target_rollout":
        agent.set_target_world_model_from_checkpoint(args.checkpoint)
    agent.world_model.eval()

    loader = get_dataloader(
        args.valid_data,
        batch_size=args.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(loader))
    images, states, actions = batch[:3]
    bboxes = batch[4] if len(batch) >= 5 else None

    metrics = agent.evaluate_transition_target_rollout_batch(
        images,
        states,
        actions,
        obstacle_bbox=bboxes,
    )
    rows, horizons = _collect_images(agent, images, states, actions, bboxes, args.max_samples)
    out_path = Path(args.output_dir) / "rollout_collapse_grid.png"
    _save_grid(rows, horizons, out_path)

    print(f"checkpoint: {args.checkpoint}")
    print(f"valid_data: {args.valid_data}")
    print(f"output: {out_path}")
    print(f"horizons: {horizons}")
    for key in [
        "gt_future_var",
        "target_recon_future_var",
        "online_recon_future_var",
        "rollout_future_var",
        "rollout_var_ratio",
        "action_sensitivity_left_right_mse",
        "action_sensitivity_gt_left_mse",
        "action_sensitivity_gt_right_mse",
        "bbox_valid_count_raw",
        "bbox_total_count_raw",
        "bbox_valid_frac_raw",
        "obstacle_valid_count",
        "obstacle_valid_frac",
    ]:
        print(f"{key}: {metrics.get(key, 0.0)}")


if __name__ == "__main__":
    main()
