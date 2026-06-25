from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from active_inference.config import Config
from active_inference.models.encoder import ConvEncoder
from active_inference.models.decoder import ObsDecoder, StateDecoder
from active_inference.models.rssm import RSSM, RSSMState
from active_inference.models.ensemble import EnsembleTransitionHeads
from active_inference.planning.cem_planner import iCEMPlanner, PlanResult
from active_inference.planning.efe import EFEScorer
from active_inference.training.losses import compute_overshoot_kl, compute_vfe
from active_inference.training.preference import PreferenceModel
from active_inference.utils.transforms import crop_road as _crop_road


class WorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        wm_type = getattr(getattr(cfg, "model", None), "world_model_type", "rssm")
        self._wm_type = wm_type
        # Centralised preprocessing flag: crop happens in preprocess_image(),
        # not inside encoder, to ensure encoder input == reconstruction target.
        self._crop_road = getattr(cfg.encoder, "crop_road", False)

        if wm_type == "rssm":
            self.encoder = ConvEncoder(
                image_channels=cfg.encoder.image_channels,
                state_dim=cfg.encoder.state_dim,
                embed_dim=cfg.rssm.embed_dim,
                crop_road=False,  # disabled: WorldModel.preprocess_image handles it
            )
            feat_dim = cfg.rssm.deter_dim + cfg.rssm.stoch_dim
            self.obs_decoder = ObsDecoder(feat_dim=feat_dim, image_channels=cfg.encoder.image_channels)
            self.state_decoder = StateDecoder(feat_dim=feat_dim, state_dim=cfg.encoder.state_dim)
            self.rssm = RSSM(
                stoch_dim=cfg.rssm.stoch_dim,
                deter_dim=cfg.rssm.deter_dim,
                embed_dim=cfg.rssm.embed_dim,
                action_dim=cfg.cem.action_dim,
                min_std=cfg.rssm.min_std,
                logvar_clip_low=cfg.rssm.logvar_clip_low,
                logvar_clip_high=cfg.rssm.logvar_clip_high,
            )

        elif wm_type == "token_vit":
            from active_inference.models.token_vit import (
                TokenViTEncoder,
                TokenViTTransition,
                TokenImageDecoder,
                TokenStateDecoder,
            )
            tv = cfg.token_vit
            feat_dim = tv.feat_dim  # deter_dim + stoch_dim = 320
            self.encoder = TokenViTEncoder(
                image_size=tv.image_size,
                patch_size=tv.patch_size,
                embed_dim=tv.embed_dim,
                state_dim=cfg.encoder.state_dim,
                num_layers=tv.num_layers,
                num_heads=tv.num_heads,
                mlp_ratio=tv.mlp_ratio,
                dropout=tv.dropout,
            )
            self.rssm = TokenViTTransition(
                num_tokens=tv.num_tokens,
                embed_dim=tv.embed_dim,
                deter_dim=tv.deter_dim,
                stoch_dim=tv.stoch_dim,
                action_dim=getattr(tv, "action_dim", cfg.cem.action_dim),
                num_prior_layers=tv.num_prior_layers,
                num_post_layers=tv.num_post_layers,
                num_heads=tv.num_heads,
                mlp_ratio=tv.mlp_ratio,
                min_std=tv.min_std,
                use_action_warp=getattr(tv, "use_action_warp", False),
            )
            self.obs_decoder = TokenImageDecoder(
                num_tokens=tv.num_tokens,
                deter_dim=tv.deter_dim,
                stoch_dim=tv.stoch_dim,
                image_channels=cfg.encoder.image_channels,
            )
            self.state_decoder = TokenStateDecoder(
                feat_dim=feat_dim,
                state_dim=cfg.encoder.state_dim,
            )

        else:
            raise ValueError(f"Unknown world_model_type: {wm_type!r}")

        self.ensemble = EnsembleTransitionHeads(
            feat_dim=feat_dim,
            stoch_dim=cfg.rssm.stoch_dim if wm_type == "rssm" else cfg.token_vit.stoch_dim,
            hidden_dim=cfg.ensemble.hidden_dim,
            num_heads=cfg.ensemble.num_heads,
        )
        # Auxiliary obstacle prediction head (training only).
        # Forces the latent space to encode obstacle presence so
        # that imagination can distinguish obstacle-present from
        # obstacle-free trajectories. Not used at eval time.
        self.obstacle_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def decode_obs(self, post: RSSMState) -> "torch.Tensor":
        """Decode observation from posterior state, routing by model type."""
        if self._wm_type == "rssm":
            return self.obs_decoder(self.rssm.get_feat(post))
        # token_vit: decoder accepts 3D (B,N,D) and (B,N,Z) directly.
        return self.obs_decoder(post.deter, post.stoch)

    def get_kl_stats(self, state: RSSMState) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return (mean, std) for VFE KL computation, token-wise when available."""
        if self._wm_type == "token_vit":
            return state.token_mean, state.token_std  # (B, N, Z)
        return state.mean, state.std                  # (B, Z)

    def preprocess_image(self, img: "torch.Tensor") -> "torch.Tensor":
        """Return the model-space image (crop_road applied if configured)."""
        if self._crop_road:
            return _crop_road(img)
        return img

    def encode_obs(self, img: "torch.Tensor", state: "torch.Tensor") -> "torch.Tensor":
        """Preprocess raw image then run encoder. Use instead of encoder() directly."""
        return self.encoder(self.preprocess_image(img), state)


class DeepAIFAgent:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.world_model = WorldModel(cfg).to(self._device)
        self.preference = PreferenceModel(
            K=cfg.preference.K,
            latent_dim=cfg.rssm.stoch_dim,
            min_std=cfg.preference.min_std,
        ).to(self._device)
        self.planner = iCEMPlanner(
            action_dim=cfg.cem.action_dim,
            horizon=cfg.cem.horizon,
            n_samples=cfg.cem.n_samples,
            n_elites=cfg.cem.n_elites,
            n_iters=cfg.cem.n_iters,
            colored_noise_beta=cfg.cem.colored_noise_beta,
            warm_start=cfg.cem.warm_start,
            noise_scale=cfg.cem.noise_scale,
            accel_prior=cfg.cem.accel_prior,
            cold_start_extra_iters=cfg.cem.cold_start_extra_iters,
            min_std=cfg.cem.min_std,
            keep_fraction=cfg.cem.keep_fraction,
            warm_start_reset_threshold=getattr(cfg.cem, "warm_start_reset_threshold", 1.0),
        )
        self.efe_scorer = EFEScorer(
            beta_instrumental=cfg.efe.beta_instrumental,
            beta_epistemic=cfg.efe.beta_epistemic,
            beta_state=cfg.efe.beta_state,
            beta_obstacle=getattr(cfg.efe, "beta_obstacle", 0.0),
            mc_samples=cfg.efe.mc_samples,
            temporal_discount=cfg.efe.temporal_discount,
            heading_only_state=getattr(cfg.efe, "heading_only_state", False),
        )

        # Two-stage training: "joint" (default), "ae" (encoder-decoder only),
        # or "transition" (rssm only, encoder-decoder frozen).
        self._stage = getattr(cfg.training, "stage", "joint")
        if self._stage == "transition":
            self.freeze_encoder_decoder()
            self._optimizer = torch.optim.Adam(
                self.world_model.rssm.parameters(), lr=cfg.training.lr,
            )
        elif self._stage == "ae":
            wm = self.world_model
            ae_params = (
                list(wm.encoder.parameters())
                + list(wm.obs_decoder.parameters())
                + list(wm.state_decoder.parameters())
            )
            self._optimizer = torch.optim.Adam(ae_params, lr=cfg.training.lr)
        else:  # joint — original behaviour
            self._optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=cfg.training.lr,
            )

        # Mixed precision: prefer bf16 (no GradScaler, avoids fp16 overflow that
        # produces NaN on large rollout-recon values); fall back to fp16+scaler.
        self._use_amp = "cuda" in str(self._device)
        if self._use_amp and torch.cuda.is_bf16_supported():
            self._amp_dtype = torch.bfloat16
            self._scaler = None  # bf16 has fp32 exponent range; no loss scaling
        elif self._use_amp:
            self._amp_dtype = torch.float16
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._amp_dtype = torch.float32
            self._scaler = None
        self._prev_state: RSSMState | None = None
        self._prev_action: Tensor | None = None
        self._scheduler = None

    def attach_scheduler(self, total_steps: int):
        """Build a warmup→cosine-decay LR schedule on the current optimizer.

        Steps are advanced inside update()/update_ae() after optimizer.step().
        No-op when training.warmup_steps <= 0 (constant LR — regression).
        """
        import math as _math
        from torch.optim.lr_scheduler import LambdaLR

        warmup = int(getattr(self._cfg.training, "warmup_steps", 0))
        if warmup <= 0:
            self._scheduler = None
            return None
        min_ratio = float(getattr(self._cfg.training, "min_lr_ratio", 0.0333))
        total_steps = max(total_steps, warmup + 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            prog = (step - warmup) / max(1, total_steps - warmup)
            prog = min(1.0, prog)
            cos = 0.5 * (1.0 + _math.cos(_math.pi * prog))
            return min_ratio + (1.0 - min_ratio) * cos

        self._scheduler = LambdaLR(self._optimizer, lr_lambda)
        return self._scheduler

    def freeze_encoder_decoder(self):
        """Freeze encoder + obs_decoder + state_decoder (stage-2 transition).

        Sets requires_grad=False and puts the modules in eval() so dropout/BN
        (if any) behave deterministically.  The RSSM transition stays trainable.
        """
        wm = self.world_model
        for mod in (wm.encoder, wm.obs_decoder, wm.state_decoder):
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()

    def reset(self):
        self._prev_state = self.world_model.rssm.initial(1, self._device)
        self._prev_action = torch.zeros(1, self._cfg.cem.action_dim, device=self._device)
        self.planner.reset()

    @torch.no_grad()
    def step_with_info(
        self,
        obs_img: Tensor,
        obs_state: Tensor,
        obstacle_info: dict | None = None,
    ) -> PlanResult:
        if self._prev_state is None:
            self.reset()

        img = (
            obs_img.unsqueeze(0).to(self._device)
            if obs_img.dim() == 3
            else obs_img.to(self._device)
        )
        st = (
            obs_state.unsqueeze(0).to(self._device)
            if obs_state.dim() == 1
            else obs_state.to(self._device)
        )

        img_model = self.world_model.preprocess_image(img)
        embed = self.world_model.encoder(img_model, st)
        post, _ = self.world_model.rssm.obs_step(
            self._prev_state, self._prev_action, embed,
        )

        # AIF precision-weighting: observed state error modulates
        # action prior precision and EFE channel weights.
        heading_err = float(st[0, 2]) if st.dim() == 2 else float(st[2])
        crosstrack_err = (
            float(st[0, 3]) if st.dim() == 2 else float(st[3])
        )
        state_error = heading_err ** 2 + crosstrack_err ** 2
        self.planner.set_state_context(
            heading_err, crosstrack_err, state_error,
        )

        # Adaptive EFE precision: high state error → trust state decoder
        # over preference model (which is OOD for off-center states).
        if state_error > 0.3:
            boost = min(4.0, 1.0 + (state_error - 0.3) * 4.0)
            self.efe_scorer._beta_s = self._cfg.efe.beta_state * boost
            self.efe_scorer._beta_i = (
                self._cfg.efe.beta_instrumental / boost
            )
        else:
            self.efe_scorer._beta_s = self._cfg.efe.beta_state
            self.efe_scorer._beta_i = self._cfg.efe.beta_instrumental

        # Visual surprise: compare decoded image against preprocessed model-space
        # image (same space as decoder output). If crop_road=true, ref_image is
        # road-cropped and resized, not the raw full frame.
        feat = self.world_model.rssm.get_feat(post)
        recon_img = self.world_model.decode_obs(post)
        recon_error = float(
            (recon_img - img_model).pow(2).sum(dim=(1, 2, 3)).item()
        )
        efe_obs_decoder = (
            self.world_model.obs_decoder
            if self.world_model._wm_type == "rssm"
            else self.world_model.obs_decoder.decode_from_feat
        )
        visual_info = {
            "recon_error": recon_error,
            "ref_image": img_model.detach(),
            "obs_decoder": efe_obs_decoder,
        }

        plan_result = self.planner.plan(
            post,
            self.world_model.rssm,
            self.efe_scorer,
            self.preference,
            self.world_model.ensemble,
            self.world_model.state_decoder,
            visual_info,
        )

        self._prev_state = post
        self._prev_action = plan_result.action.unsqueeze(0)
        return plan_result

    @torch.no_grad()
    def step(
        self,
        obs_img: Tensor,
        obs_state: Tensor,
        obstacle_info: dict | None = None,
    ) -> Tensor:
        return self.step_with_info(obs_img, obs_state).action

    @staticmethod
    def _flip_bbox_x(bbox: Tensor, image_size: int = 64) -> Tensor:
        """Mirror bbox xyxy x-coords: x_new = image_size - x_old.

        The obstacle_bbox is stored in CARLA camera coords (X = right), which is
        left-right mirrored relative to image columns (col 0 = left).  Applying
        this flip aligns bbox x with the actual obstacle position in the image.
        Preserves NaN rows (NaN - x = NaN) so downstream NaN handling still works.
        y-coords are unchanged.
        """
        out = bbox.clone()
        out[..., 0] = image_size - bbox[..., 2]   # new x1 = img_size - old x2
        out[..., 2] = image_size - bbox[..., 0]   # new x2 = img_size - old x1
        return out

    @staticmethod
    def _bbox_weight_map(
        bbox: Tensor, img_shape, obstacle_weight: float, device,
    ) -> Tensor:
        """Build a (B,1,H,W) weight map: 1 outside bbox, obstacle_weight inside.

        bbox: (B, 4) pixel xyxy. NaN/invalid rows produce all-ones (uniform).
        img_shape: (B, C, H, W) of the reconstruction.
        """
        B, _, H, W = img_shape
        w_map = torch.ones(B, 1, H, W, device=device)
        bbox = bbox.to(device).float()
        # Correct CARLA camera x-mirror so the box matches the image columns.
        bbox = DeepAIFAgent._flip_bbox_x(bbox, W)
        ys = torch.arange(H, device=device).view(1, H, 1)
        xs = torch.arange(W, device=device).view(1, 1, W)
        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            if torch.isnan(bbox[b]).any() or x2 <= x1 or y2 <= y1:
                continue
            inside_y = (ys[0] >= y1) & (ys[0] <= y2)   # (H,1)
            inside_x = (xs[0] >= x1) & (xs[0] <= x2)   # (1,W)
            mask = inside_y & inside_x                  # (H,W)
            w_map[b, 0] = 1.0 + (obstacle_weight - 1.0) * mask.float()
        return w_map

    @staticmethod
    def _bbox_token_mask(
        bbox: Tensor, num_tokens: int, image_size: int, patch_size: int, device,
        crop_road: bool = False, keep_bottom_frac: float = 0.6,
    ) -> Tensor:
        """Map xyxy-pixel bbox to a (B, num_tokens) {0,1} token-overlap mask.

        Token i covers image region
            rows [(i//G)*P : (i//G+1)*P], cols [(i%G)*P : (i%G+1)*P]
        with G = image_size//patch_size (=8 for 64/8).  Matches the raster
        token order used by the ViT encoder/decoder.  NaN/degenerate bbox → all
        zeros (no obstacle tokens → uniform weighting downstream).

        crop_road: the encoder sees crop_road(keep_bottom_frac) — top
        (1-keep_bottom_frac) is dropped and the bottom band is stretched to
        image_size.  bbox y-coords (in ORIGINAL pixels) are remapped into this
        cropped space so tokens align with what the encoder/decoder actually
        process.  crop_road only affects rows.  A bbox entirely above the kept
        band degenerates → mask 0.

        x-coords are first mirrored (CARLA X=right vs image col 0=left) via
        _flip_bbox_x so the box lands on the correct columns.
        """
        B = bbox.shape[0]
        G = image_size // patch_size
        P = patch_size
        mask = torch.zeros(B, num_tokens, device=device)
        bbox = bbox.to(device).float()
        # Correct CARLA camera x-mirror (x only; y handled by crop remap below).
        bbox = DeepAIFAgent._flip_bbox_x(bbox, image_size)

        start = image_size * (1.0 - keep_bottom_frac)  # first kept row (orig)
        crop_h = image_size - start                    # kept band height

        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            if torch.isnan(bbox[b]).any() or x2 <= x1 or y2 <= y1:
                continue
            if crop_road:
                # Remap y into the cropped+resized model space (x unchanged).
                y1 = (y1 - start) * image_size / crop_h
                y2 = (y2 - start) * image_size / crop_h
                y1 = max(0.0, float(y1))
                y2 = min(float(image_size), float(y2))
                if y2 <= y1:
                    continue  # obstacle fell outside the kept band
            for i in range(num_tokens):
                r, c = i // G, i % G
                ty1, ty2 = r * P, (r + 1) * P
                tx1, tx2 = c * P, (c + 1) * P
                # overlap test (token region vs bbox)
                if tx1 < x2 and tx2 > x1 and ty1 < y2 and ty2 > y1:
                    mask[b, i] = 1.0
        return mask

    def update(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_labels: Tensor | None = None,
        obstacle_bbox: Tensor | None = None,
    ) -> dict[str, float]:
        # Per-timestep backward with NaN guard, AMP, and CUDA error recovery
        # obstacle_labels: [B, T] binary (1=obstacle visible, 0=clear)
        # obstacle_bbox:   [B, T, 4] pixel xyxy bbox; None = no bbox weighting
        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
        cfg = self._cfg.training
        beta_obstacle_aux = getattr(cfg, "beta_obstacle_aux", 0.0)

        overshoot_horizon = getattr(cfg, "overshoot_horizon", 0)
        overshoot_weight = getattr(cfg, "overshoot_weight", 0.0)
        token_kl_weighting = getattr(cfg, "token_kl_weighting", "none")
        warp_smoothness_weight = getattr(cfg, "warp_smoothness_weight", 0.0)
        rollout_recon_horizon = getattr(cfg, "rollout_recon_horizon", 0)
        rollout_recon_weight = getattr(cfg, "rollout_recon_weight", 0.0)
        rollout_recon_obs_w = getattr(cfg, "rollout_recon_obstacle_weight", 1.0)
        rollout_recon_decay = getattr(cfg, "rollout_recon_step_decay", 0.0)
        img_recon_obs_w = getattr(cfg, "img_recon_obstacle_weight", 1.0)
        cycle_weight = getattr(cfg, "cycle_weight", 0.0)
        obstacle_token_weight = getattr(cfg, "obstacle_token_weight", 1.0)
        # Cycle loss runs along the rollout up to cycle_horizon (default 5),
        # independent of overshoot_horizon (which defaults to 0).  The unified
        # rollout loop must therefore span the max of all three horizons.
        cycle_horizon = getattr(cfg, "cycle_horizon", 5)
        rr_horizon_eff = max(rollout_recon_horizon, overshoot_horizon, cycle_horizon)
        # ViT token-grid params for obstacle-token weighting in the cycle loss.
        _tv = getattr(self._cfg, "token_vit", None)
        _is_vit = wm._wm_type == "token_vit"
        use_token_weight = (
            obstacle_token_weight > 1.0 and _is_vit
            and obstacle_bbox is not None
        )
        # crop_road remaps bbox y-coords so token mask aligns with encoder input.
        _crop = bool(getattr(self._cfg.encoder, "crop_road", False))
        _keep_frac = float(getattr(self._cfg.encoder, "keep_bottom_frac", 0.6))

        self._optimizer.zero_grad()
        total_loss_value = 0.0
        accum = {
            "img_loss": 0.0, "state_loss": 0.0,
            "kl_dyn": 0.0, "kl_rep": 0.0, "obs_aux_loss": 0.0,
            "overshoot_kl": 0.0, "rollout_recon": 0.0, "cycle": 0.0,
        }

        # ── 1st pass: collect stop-grad posterior references ──────────────────
        # Runs encoder + obs_step once under no_grad to build the target list.
        # posteriors_ref[t].mean/.std are fixed targets for overshoot KL and the
        # cycle loss (rollout prior mean vs posterior mean).
        posteriors_ref: list[RSSMState] = []
        _need_ref = (
            (overshoot_horizon > 0 and overshoot_weight > 0.0)
            or (cycle_weight > 0.0)
        )
        if _need_ref:
            prev_s = wm.rssm.initial(B, self._device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                    for t in range(T):
                        img_ref = wm.preprocess_image(images[:, t].to(self._device))
                        st_ref = states[:, t].to(self._device)
                        act_ref = actions[:, t].to(self._device)
                        emb_ref = wm.encoder(img_ref, st_ref)
                        post_ref, _ = wm.rssm.obs_step(prev_s, act_ref, emb_ref)
                        posteriors_ref.append(post_ref)
                        prev_s = type(post_ref)(*[x.detach() for x in post_ref])

        # ── 2nd pass: main training loop ──────────────────────────────────────
        # Dataset convention used by the v5 collectors:
        #   action[t] is the action applied by env.step(action[t]) that produced
        #   image[t], state[t]. Therefore obs_step at timestep t must be
        #   conditioned on actions[:, t], not on a one-step-delayed prev_action.
        prev_state = wm.rssm.initial(B, self._device)

        for t in range(T):
            img_raw_t = images[:, t].to(self._device)
            img_t = wm.preprocess_image(img_raw_t)  # encoder input == recon target
            st_t = states[:, t].to(self._device)
            act_t = actions[:, t].to(self._device)

            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(
                    prev_state, act_t, embed,
                )

                feat = wm.rssm.get_feat(post)
                recon_img = wm.decode_obs(post)
                recon_state = wm.state_decoder(feat)

                post_mean, post_std = wm.get_kl_stats(post)
                prior_mean, prior_std = wm.get_kl_stats(prior)
                bbox_t = (
                    obstacle_bbox[:, t].to(self._device)
                    if (obstacle_bbox is not None and img_recon_obs_w > 1.0)
                    else None
                )
                loss, info = compute_vfe(
                    post_mean,
                    post_std,
                    prior_mean,
                    prior_std,
                    img_t,       # preprocessed image — same space as decoder output
                    recon_img,
                    st_t,
                    recon_state,
                    free_nats=cfg.free_nats,
                    kl_dyn_scale=cfg.kl_dyn_scale,
                    kl_rep_scale=cfg.kl_rep_scale,
                    token_kl_weighting=token_kl_weighting,
                    obstacle_bbox=bbox_t,
                    img_recon_obstacle_weight=img_recon_obs_w,
                )

                # Stage-2 transition: encoder/decoders are frozen, so img_loss
                # and state_loss must not drive the backward pass.  Subtracting
                # the (same) tensors removes their gradient contribution exactly,
                # leaving only the KL terms (+ overshoot + rollout_recon below).
                # img_loss/state_loss remain in `info` for monitoring/logging.
                if self._stage == "transition":
                    loss = loss - info["img_loss"] - info["state_loss"]

                # A1: warp smoothness regularization (set by last img_step call)
                if warp_smoothness_weight > 0 and hasattr(wm.rssm, "warp_smoothness_loss"):
                    smooth = wm.rssm.warp_smoothness_loss()
                    loss = loss + warp_smoothness_weight * smooth

                # Multi-step pixel rollout reconstruction loss.
                # From the current posterior, roll out H prior steps using GT actions
                # and compare decoded images to GT future frames.  Trains the prior
                # decoder to produce visually coherent futures.
                rr_val = 0.0
                cycle_val = 0.0
                do_rr = rollout_recon_horizon > 0 and rollout_recon_weight > 0.0
                do_cycle = cycle_weight > 0.0 and bool(posteriors_ref)
                if (do_rr or do_cycle) and t + 1 < T:
                    n_rr = min(rr_horizon_eff, T - 1 - t)
                    if n_rr > 0:
                        state_rr = post
                        rr_loss    = torch.zeros((), device=self._device)
                        cycle_loss = torch.zeros((), device=self._device)
                        n_cyc = 0
                        for h in range(1, n_rr + 1):
                            act_rr    = actions[:, t + h].to(self._device)
                            state_rr  = wm.rssm.img_step(state_rr, act_rr)

                            # Cycle loss: rollout prior mean ↔ stop-grad posterior
                            # mean at t+h.  Anchors the open-loop prior to the
                            # filtered (posterior) trajectory → stabilizes rollout.
                            # For ViT, obstacle-overlap tokens are upweighted so
                            # the prior matches the latent where the obstacle is.
                            if do_cycle and h <= cycle_horizon and t + h < len(posteriors_ref):
                                pm, _ = wm.get_kl_stats(state_rr)
                                qm, _ = wm.get_kl_stats(posteriors_ref[t + h])
                                sq = (pm - qm.detach()).pow(2)  # (B,N,Z) ViT | (B,Z) RSSM
                                if use_token_weight and pm.ndim == 3:
                                    tok_mask = self._bbox_token_mask(
                                        obstacle_bbox[:, t + h], pm.shape[1],
                                        _tv.image_size, _tv.patch_size, self._device,
                                        crop_road=_crop, keep_bottom_frac=_keep_frac,
                                    )  # (B, N)
                                    w_tok = 1.0 + (obstacle_token_weight - 1.0) * tok_mask
                                    sq = sq * w_tok.unsqueeze(-1)  # (B,N,1) broadcast
                                cycle_loss = cycle_loss + sq.mean()
                                n_cyc += 1

                            if do_rr and h <= rollout_recon_horizon:
                                recon_rr  = wm.decode_obs(state_rr)
                                target_rr = wm.preprocess_image(
                                    images[:, t + h].to(self._device)
                                )
                                mse_rr = (recon_rr - target_rr).pow(2)  # (B,C,H,W)
                                if rollout_recon_obs_w > 1.0 and obstacle_bbox is not None:
                                    w_map = self._bbox_weight_map(
                                        obstacle_bbox[:, t + h],
                                        recon_rr.shape,
                                        rollout_recon_obs_w,
                                        self._device,
                                    )  # (B,1,H,W)
                                    mse_rr = mse_rr * w_map
                                step_w = (1.0 - rollout_recon_decay) ** (h - 1)
                                rr_loss = rr_loss + step_w * mse_rr.mean()

                        if do_rr:
                            n_rr_used = min(rollout_recon_horizon, T - 1 - t)
                            if n_rr_used > 0:
                                rr_loss = rr_loss / n_rr_used
                                loss    = loss + rollout_recon_weight * rr_loss
                                rr_val  = rr_loss.item()
                        if do_cycle and n_cyc > 0:
                            cycle_loss = cycle_loss / n_cyc
                            loss = loss + cycle_weight * cycle_loss
                            cycle_val = cycle_loss.item()

                # Auxiliary obstacle prediction loss
                obs_aux_loss_val = 0.0
                if (obstacle_labels is not None
                        and beta_obstacle_aux > 0):
                    obs_logit = wm.obstacle_head(feat).squeeze(-1)
                    obs_label = obstacle_labels[:, t].to(
                        self._device,
                    ).float()
                    obs_aux = nn.functional.binary_cross_entropy_with_logits(
                        obs_logit, obs_label,
                    )
                    loss = loss + beta_obstacle_aux * obs_aux
                    obs_aux_loss_val = obs_aux.item()

                # Latent overshooting: D-step prior rollout vs stop-grad posteriors.
                # wm.get_kl_stats() dispatches to (B,Z) for RSSM or (B,N,Z) for ViT,
                # so compute_overshoot_kl works for both model types.
                # Guard overshoot_horizon > 0: posteriors_ref may be populated
                # solely for the cycle loss (overshoot off), in which case
                # n_steps would be 0 and osh_kl/n_steps = 0/0 = NaN, poisoning
                # the whole step's loss → NaN-guard skip → cycle/rollout dropped.
                osh_kl_val = 0.0
                if posteriors_ref and overshoot_horizon > 0 and t + 1 < T:
                    n_steps = min(overshoot_horizon, T - 1 - t)
                    state_d = post
                    osh_kl = torch.zeros((), device=self._device)
                    for d in range(1, n_steps + 1):
                        act_td = actions[:, t + d].to(self._device)
                        state_d = wm.rssm.img_step(state_d, act_td)
                        ref = posteriors_ref[t + d]
                        osh_kl = osh_kl + compute_overshoot_kl(
                            *wm.get_kl_stats(state_d),
                            *wm.get_kl_stats(ref),
                            free_nats=cfg.free_nats,
                        )
                    osh_kl = osh_kl / n_steps
                    loss = loss + overshoot_weight * osh_kl
                    osh_kl_val = osh_kl.item()

            # NaN guard: skip this timestep if loss is bad
            if torch.isnan(loss) or torch.isinf(loss):
                prev_state = type(post)(
                    *[x.detach() for x in post],
                )
                continue

            scaled_loss = loss / T
            if self._scaler is not None:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss_value += loss.item() / T

            for k in ["img_loss", "state_loss", "kl_dyn", "kl_rep"]:
                accum[k] += info[k].item()
            accum["obs_aux_loss"] += obs_aux_loss_val
            accum["overshoot_kl"] += osh_kl_val
            accum["rollout_recon"] += rr_val
            accum["cycle"] += cycle_val

            prev_state = type(post)(
                *[x.detach() for x in post],
            )

        # Clip only the optimized parameters (rssm-only in transition stage).
        clip_params = self._optimizer.param_groups[0]["params"]
        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            self._optimizer.step()

        # LR scheduler step (set by train.py via attach_scheduler), if any.
        sched = getattr(self, "_scheduler", None)
        if sched is not None:
            sched.step()

        return {k: v / T for k, v in accum.items()} | {"total_loss": total_loss_value}

    @staticmethod
    def _det_state(post):
        """Return a copy of an RSSM/Token state with stoch replaced by its mean.

        Removes sampling noise so reconstruction is deterministic.
        """
        if hasattr(post, "token_mean"):
            return type(post)(deter=post.deter, stoch=post.token_mean,
                             mean=post.mean, std=post.std,
                             token_mean=post.token_mean, token_std=post.token_std)
        return type(post)(deter=post.deter, stoch=post.mean,
                         mean=post.mean, std=post.std)

    def update_ae(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_labels: Tensor | None = None,
        obstacle_bbox: Tensor | None = None,
    ) -> dict[str, float]:
        """Stage-1 autoencoder update: train encoder + obs_decoder + state_decoder.

        Posterior is obtained via obs_step (same as update) but decoded
        deterministically (stoch = mean); the prior prediction is unused.
        Loss = img_loss + state_loss + ae_kl_rep * KL(posterior ‖ N(0,1)).
        The optimizer (built in __init__ for stage="ae") contains only the
        encoder + decoders, so the RSSM transition is not trained here.
        """
        from torch.distributions import Normal, kl_divergence

        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
        cfg = self._cfg.training
        ae_kl_rep = getattr(cfg, "ae_kl_rep", 0.0)
        img_recon_obs_w = getattr(cfg, "img_recon_obstacle_weight", 1.0)

        self._optimizer.zero_grad()
        total_loss_value = 0.0
        accum = {"img_loss": 0.0, "state_loss": 0.0, "kl_rep": 0.0}

        prev_state = wm.rssm.initial(B, self._device)

        for t in range(T):
            img_t = wm.preprocess_image(images[:, t].to(self._device))
            st_t  = states[:, t].to(self._device)
            act_t = actions[:, t].to(self._device)

            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                embed = wm.encoder(img_t, st_t)
                post, _prior = wm.rssm.obs_step(prev_state, act_t, embed)

                det_post   = self._det_state(post)
                feat       = wm.rssm.get_feat(det_post)
                recon_img  = wm.decode_obs(det_post)
                recon_state = wm.state_decoder(feat)

                # Image reconstruction (with optional obstacle bbox upweight)
                err = (recon_img - img_t).pow(2)  # (B,C,H,W)
                if img_recon_obs_w > 1.0 and obstacle_bbox is not None:
                    w_map = self._bbox_weight_map(
                        obstacle_bbox[:, t].to(self._device), recon_img.shape,
                        img_recon_obs_w, self._device,
                    )
                    err = err * w_map
                c, h, w = img_t.shape[1], img_t.shape[2], img_t.shape[3]
                img_loss = err.sum(dim=(1, 2, 3)).mean() / (c * h * w)

                from active_inference.utils.transforms import symlog
                state_loss = nn.functional.mse_loss(
                    symlog(recon_state), symlog(st_t),
                )

                # Weak VAE-style posterior regularization toward N(0,1)
                p_mean, p_std = wm.get_kl_stats(post)
                kl_rep = kl_divergence(
                    Normal(p_mean, p_std),
                    Normal(torch.zeros_like(p_mean), torch.ones_like(p_std)),
                ).sum(-1).mean()

                loss = img_loss + state_loss + ae_kl_rep * kl_rep

            if torch.isnan(loss) or torch.isinf(loss):
                prev_state = type(post)(*[x.detach() for x in post])
                continue

            scaled_loss = loss / T
            if self._scaler is not None:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss_value += loss.item() / T
            accum["img_loss"]   += img_loss.item() / T
            accum["state_loss"] += state_loss.item() / T
            accum["kl_rep"]     += kl_rep.item() / T

            prev_state = type(post)(*[x.detach() for x in post])

        clip_params = self._optimizer.param_groups[0]["params"]
        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
            self._optimizer.step()

        sched = getattr(self, "_scheduler", None)
        if sched is not None:
            sched.step()

        return accum | {"total_loss": total_loss_value}

    @torch.no_grad()
    def evaluate_ae_batch(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[dict[str, float], int]:
        """Compute AE-stage losses on one batch with the SAME forward path as
        update_ae (deterministic decode), without backward.

        Returns (sum_over_valid_steps dict, valid_steps).  The caller averages.
        Critically mirrors update_ae so validation img/state reflect the actual
        reconstruction quality the AE stage optimizes — not the untrained
        transition path used by the joint/transition evaluator.
        """
        from torch.distributions import Normal, kl_divergence
        from active_inference.utils.transforms import symlog

        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
        prev_state = wm.rssm.initial(B, self._device)

        acc = {"img_loss": 0.0, "state_loss": 0.0, "kl_rep": 0.0}
        valid_steps = 0

        for t in range(T):
            img_t = wm.preprocess_image(images[:, t].to(self._device))
            st_t  = states[:, t].to(self._device)
            act_t = actions[:, t].to(self._device)

            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                embed = wm.encoder(img_t, st_t)
                post, _prior = wm.rssm.obs_step(prev_state, act_t, embed)

                det_post    = self._det_state(post)
                feat        = wm.rssm.get_feat(det_post)
                recon_img   = wm.decode_obs(det_post)
                recon_state = wm.state_decoder(feat)

                c, h, w = img_t.shape[1], img_t.shape[2], img_t.shape[3]
                img_loss = (recon_img - img_t).pow(2).sum(dim=(1, 2, 3)).mean() / (c * h * w)
                state_loss = nn.functional.mse_loss(symlog(recon_state), symlog(st_t))

                p_mean, p_std = wm.get_kl_stats(post)
                kl_rep = kl_divergence(
                    Normal(p_mean, p_std),
                    Normal(torch.zeros_like(p_mean), torch.ones_like(p_std)),
                ).sum(-1).mean()

            if torch.isnan(img_loss) or torch.isinf(img_loss):
                prev_state = type(post)(*[x.detach() for x in post])
                continue

            acc["img_loss"]   += img_loss.item()
            acc["state_loss"] += state_loss.item()
            acc["kl_rep"]     += kl_rep.item()
            valid_steps += 1
            prev_state = type(post)(*[x.detach() for x in post])

        return acc, valid_steps

    def save_checkpoint(self, path: str | Path):
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "preference": {
                    "means": self.preference.means.data,
                    "log_stds": self.preference.log_stds.data,
                    "logits": self.preference.logits.data,
                },
            },
            path,
        )

    def load_encoder_decoder(self, path: str | Path):
        """Load ONLY encoder + obs_decoder + state_decoder weights from a
        stage-1 (ae) checkpoint.  Leaves the RSSM transition and the optimizer
        untouched — used by stage-2 (transition) training via --init_from.
        """
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        full_sd = ckpt["world_model"]
        wm = self.world_model
        loaded = []
        for prefix, module in (
            ("encoder.", wm.encoder),
            ("obs_decoder.", wm.obs_decoder),
            ("state_decoder.", wm.state_decoder),
        ):
            sub = {k[len(prefix):]: v for k, v in full_sd.items()
                   if k.startswith(prefix)}
            if sub:
                module.load_state_dict(sub)
                loaded.append(prefix.rstrip("."))
        print(f"  Loaded encoder-decoder from {path}: {loaded}")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        surgery_applied = False
        try:
            self.world_model.load_state_dict(ckpt["world_model"])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"State dim mismatch detected, applying weight surgery...")
                self._load_with_state_dim_surgery(ckpt["world_model"])
                surgery_applied = True
            else:
                raise
        if surgery_applied:
            # Optimizer state has old parameter shapes; reinitialize
            print("  Reinitializing optimizer (parameter shapes changed)")
            self._optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=self._cfg.training.lr
            )
        else:
            self._optimizer.load_state_dict(ckpt["optimizer"])
        pref = ckpt["preference"]
        ckpt_K = pref["means"].shape[0]
        model_K = self.preference.means.shape[0]
        if ckpt_K == model_K:
            self.preference.means.data.copy_(pref["means"])
            self.preference.log_stds.data.copy_(pref["log_stds"])
            self.preference.logits.data.copy_(pref["logits"])
        else:
            # K mismatch: load what fits, preference will be refit later
            k = min(ckpt_K, model_K)
            self.preference.means.data[:k].copy_(pref["means"][:k])
            self.preference.log_stds.data[:k].copy_(pref["log_stds"][:k])
            self.preference.logits.data[:k].copy_(pref["logits"][:k])
            print(f"  Preference K mismatch: {ckpt_K} -> {model_K} (partial load, refit needed)")

        # Load contrastive preference if available — replaces standard GMM
        if "contrastive_preference" in ckpt:
            from active_inference.training.preference import (
                ContrastivePreferenceModel,
            )
            cp = ckpt["contrastive_preference"]
            contrastive = ContrastivePreferenceModel(
                K_clean=cp["K_clean"],
                K_avoid=cp["K_avoid"],
                latent_dim=self._cfg.rssm.stoch_dim,
                contrast_scale=cp["contrast_scale"],
            ).to(self._device)
            contrastive.clean.means.data.copy_(cp["clean_means"])
            contrastive.clean.log_stds.data.copy_(cp["clean_log_stds"])
            contrastive.clean.logits.data.copy_(cp["clean_logits"])
            contrastive.avoid.means.data.copy_(cp["avoid_means"])
            contrastive.avoid.log_stds.data.copy_(cp["avoid_log_stds"])
            contrastive.avoid.logits.data.copy_(cp["avoid_logits"])
            # Replace preference with contrastive model (same log_prob API)
            self.preference = contrastive
            print(
                f"  Loaded contrastive preference "
                f"(K_clean={cp['K_clean']}, K_avoid={cp['K_avoid']}, "
                f"scale={cp['contrast_scale']})"
            )

    def _load_with_state_dim_surgery(self, ckpt_state_dict: dict):
        """Load checkpoint with mismatched state_dim by padding new dimensions."""
        model_sd = self.world_model.state_dict()
        for key in ckpt_state_dict:
            ckpt_val = ckpt_state_dict[key]
            model_val = model_sd[key]
            if ckpt_val.shape == model_val.shape:
                model_sd[key] = ckpt_val
            else:
                # Pad with zeros for new state dimensions
                padded = torch.zeros_like(model_val)
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in
                               zip(ckpt_val.shape, model_val.shape))
                padded[slices] = ckpt_val[slices]
                model_sd[key] = padded
                print(f"  Weight surgery: {key} {ckpt_val.shape} -> {model_val.shape}")
        self.world_model.load_state_dict(model_sd)

    @torch.no_grad()
    def encode_preference_data(self, pref_loader, max_samples: int = 3000) -> Tensor:
        latents = []
        wm = self.world_model
        count = 0

        for images, states, actions in pref_loader:
            B, T = images.shape[:2]
            state = wm.rssm.initial(B, self._device)

            for t in range(T):
                act_t = actions[:, t].to(self._device)
                embed = wm.encode_obs(
                    images[:, t].to(self._device),
                    states[:, t].to(self._device),
                )
                post, _ = wm.rssm.obs_step(state, act_t, embed)
                latents.append(post.mean.cpu())
                state = type(post)(*[x.detach() for x in post])

            count += B * T
            if count >= max_samples:
                break

        return torch.cat(latents, dim=0)[:max_samples].to(self._device)

    def update_preference(self, pref_loader, max_samples: int = 3000) -> dict[str, float]:
        latents = self.encode_preference_data(pref_loader, max_samples)
        self.preference.update_from_latents(
            latents,
            n_iters=self._cfg.preference.fit_iters,
            lr=self._cfg.preference.fit_lr,
        )

        # Health metrics
        log_prob = self.preference.log_prob(latents).mean().item()
        means = self.preference.means.data
        min_dist = float("inf")
        for i in range(means.shape[0]):
            for j in range(i + 1, means.shape[0]):
                d = (means[i] - means[j]).norm().item()
                if d < min_dist:
                    min_dist = d
        weights = torch.softmax(self.preference.logits.data, dim=0)

        return {
            "pref_log_prob": log_prob,
            "pref_min_component_dist": min_dist,
            "pref_max_weight": weights.max().item(),
            "pref_min_weight": weights.min().item(),
            "pref_n_samples": latents.shape[0],
        }

    @staticmethod
    def action_to_carla(
        action_2d: Tensor,
        min_throttle: float = 0.35,
        max_throttle: float = 0.55,
    ) -> tuple[float, float, float]:
        steer = float(action_2d[0].clamp(-1, 1))
        accel = float(action_2d[1].clamp(-1, 1))
        throttle = min_throttle + (max_throttle - min_throttle) * (accel + 1.0) / 2.0
        brake = 0.0
        return steer, throttle, brake
