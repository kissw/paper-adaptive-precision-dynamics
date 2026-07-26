from __future__ import annotations

import os
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
from active_inference.training.losses import (
    compute_overshoot_kl,
    compute_transition_target_kl,
    compute_vfe,
)
from active_inference.training.preference import PreferenceModel
from active_inference.utils.bbox import (
    bbox_xyxy_to_mask,
    bbox_xyxy_to_model_space,
    bbox_xyxy_to_token_mask,
)
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
        self.target_world_model: WorldModel | None = None
        self.h1_anchor_world_model: WorldModel | None = None
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
            ae_train_rssm = bool(getattr(cfg.training, "ae_train_rssm", True))
            ae_params = list(wm.encoder.parameters())
            if ae_train_rssm:
                ae_params += list(wm.rssm.parameters())
            ae_params += list(wm.obs_decoder.parameters()) + list(wm.state_decoder.parameters())
            self._optimizer = torch.optim.Adam(ae_params, lr=cfg.training.lr)
            if ae_train_rssm:
                print("AE optimizer: encoder + rssm + decoders")
            else:
                print("AE optimizer: encoder + decoders only")
        else:  # joint — original behaviour
            self._optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=cfg.training.lr,
            )

        # Mixed precision: bf16 only (never fp16).  Transition rollouts repeat
        # img_step and produce large latent values; fp16's 65504 max overflows
        # to Inf→NaN.  bf16 has the fp32 exponent range, so no overflow and no
        # GradScaler is needed.  If bf16 is unsupported, fall back to fp32 — fp16
        # is intentionally never used.
        # FORCE_FP32=1 disables bf16 autocast (full fp32) — to test whether bf16
        # is the source of any transition-stage divergence.
        force_fp32 = os.environ.get("FORCE_FP32") == "1"
        cuda_ok = "cuda" in str(self._device)
        if cuda_ok and torch.cuda.is_bf16_supported() and not force_fp32:
            self._use_amp = True
            self._amp_dtype = torch.bfloat16
        else:
            self._use_amp = False
            self._amp_dtype = torch.float32
        self._scaler = None  # never scale: bf16/fp32 both keep fp32 exponent range
        self._prev_state: RSSMState | None = None
        self._prev_action: Tensor | None = None
        self._scheduler = None
        self._train_step = 0  # optimizer steps taken (for cycle_weight warmup)

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

    @staticmethod
    def _bbox_model_mask(
        bbox: Tensor,
        img_shape,
        device,
        crop_road: bool = False,
        keep_bottom_frac: float = 0.6,
        min_area: float = 1.0,
        dilate: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Return (B,1,H,W) bbox mask and (B,) validity in model image space."""
        B, _, H, W = img_shape
        mask = torch.zeros(B, 1, H, W, device=device)
        valid = torch.zeros(B, device=device, dtype=torch.bool)
        bbox = bbox.to(device).float()
        bbox = DeepAIFAgent._flip_bbox_x(bbox, W)

        start = H * (1.0 - keep_bottom_frac)
        crop_h = H - start
        ys = torch.arange(H, device=device).view(H, 1)
        xs = torch.arange(W, device=device).view(1, W)
        dilate = max(0, int(dilate))

        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            if torch.isnan(bbox[b]).any() or x2 <= x1 or y2 <= y1:
                continue
            if crop_road:
                y1 = (y1 - start) * H / crop_h
                y2 = (y2 - start) * H / crop_h
            x1 = max(0.0, float(x1) - dilate)
            x2 = min(float(W), float(x2) + dilate)
            y1 = max(0.0, float(y1) - dilate)
            y2 = min(float(H), float(y2) + dilate)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue
            inside = (ys >= y1) & (ys < y2) & (xs >= x1) & (xs < x2)
            if inside.any():
                mask[b, 0] = inside.float()
                valid[b] = True
        return mask, valid

    def update(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_labels: Tensor | None = None,
        obstacle_bbox: Tensor | None = None,
    ) -> dict[str, float]:
        cfg = self._cfg.training
        transition_loss_mode = getattr(cfg, "transition_loss_mode", "")
        if (
            self._stage == "transition"
            and transition_loss_mode == "target_rollout"
            and self.target_world_model is not None
        ):
            return self.update_transition_target_rollout(
                images, states, actions, obstacle_bbox=obstacle_bbox,
            )
        if (
            self._stage == "transition"
            and transition_loss_mode == "dense_one_step"
        ):
            return self.update_transition_dense_one_step(images, states, actions)

        # Per-timestep backward with NaN guard, AMP, and CUDA error recovery
        # obstacle_labels: [B, T] binary (1=obstacle visible, 0=clear)
        # obstacle_bbox:   [B, T, 4] pixel xyxy bbox; None = no bbox weighting
        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
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
        # Linear cycle_weight warmup: ramp 0 → cycle_weight over warmup steps so
        # the early prior (large latent error) does not blow up the loss.
        cycle_warmup_steps = getattr(cfg, "cycle_warmup_steps", 0)
        if cycle_warmup_steps > 0:
            ramp = min(1.0, self._train_step / cycle_warmup_steps)
        else:
            ramp = 1.0
        eff_cycle_weight = cycle_weight * ramp
        # Same warmup for rollout_recon: a fresh rssm's 5-step rollout can blow
        # up before kl_dyn stabilises, so ramp the recon weight in gradually.
        rr_warmup_steps = getattr(cfg, "rollout_recon_warmup_steps", 0)
        if rr_warmup_steps > 0:
            rr_ramp = min(1.0, self._train_step / rr_warmup_steps)
        else:
            rr_ramp = 1.0
        eff_rr_weight = rollout_recon_weight * rr_ramp
        rr_horizon_eff = max(rollout_recon_horizon, overshoot_horizon, cycle_horizon)
        # Single-shot rollout: roll out rr/cycle ONCE per sequence (from the
        # posterior at frame P-1) instead of every timestep — avoids the
        # gradient accumulation across ~T rollouts that destabilizes a fresh
        # rssm.  H spans max(rollout_recon_horizon, cycle_horizon).
        rollout_single_shot = bool(getattr(cfg, "rollout_single_shot", False))
        rollout_P = int(getattr(cfg, "rollout_context_frames", 25))
        ss_H = max(rollout_recon_horizon, cycle_horizon)
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
                    # Transition stage should train the prior/img_step only.
                    # Encoder/decoders are frozen, and posterior representation
                    # should not be pulled away from the decoder-compatible AE
                    # manifold. Remove img/state/kl_rep gradients from VFE.
                    loss = (
                        loss
                        - info["img_loss"]
                        - info["state_loss"]
                        - cfg.kl_rep_scale * info["kl_rep"]
                    )

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

                # ── Single-shot rollout (PAActInf style): once per sequence ──
                # Runs only at t == P-1, from the grad-connected posterior `post`,
                # and adds its loss to THIS timestep's loss so the per-timestep
                # backward below covers it in a single graph.
                if rollout_single_shot and (do_rr or do_cycle) and t == rollout_P - 1 and t + 1 < T:
                    state_ss = type(post)(*[x.detach() for x in post])
                    rr_loss    = torch.zeros((), device=self._device)
                    cycle_loss = torch.zeros((), device=self._device)
                    n_cyc = 0
                    n_rr_ss = 0
                    for h in range(1, ss_H + 1):
                        idx = (rollout_P - 1) + h
                        if idx >= T:
                            break
                        state_ss = wm.rssm.img_step(state_ss, actions[:, idx].to(self._device))
                        if do_cycle and h <= cycle_horizon and idx < len(posteriors_ref):
                            pm, _ = wm.get_kl_stats(state_ss)
                            qm, _ = wm.get_kl_stats(posteriors_ref[idx])
                            sq = (pm - qm.detach()).pow(2)
                            if use_token_weight and pm.ndim == 3:
                                tok_mask = self._bbox_token_mask(
                                    obstacle_bbox[:, idx], pm.shape[1],
                                    _tv.image_size, _tv.patch_size, self._device,
                                    crop_road=_crop, keep_bottom_frac=_keep_frac,
                                )
                                w_tok = 1.0 + (obstacle_token_weight - 1.0) * tok_mask
                                sq = sq * w_tok.unsqueeze(-1)
                            # Single rollout/seq → no per-step explosion; keep a
                            # loose clamp only as an Inf guard.
                            cycle_loss = cycle_loss + sq.mean().clamp(max=1e4)
                            n_cyc += 1
                        if do_rr and h <= rollout_recon_horizon:
                            recon_ss  = wm.decode_obs(state_ss)
                            target_ss = wm.preprocess_image(images[:, idx].to(self._device))
                            mse_ss = (recon_ss - target_ss).pow(2)
                            if rollout_recon_obs_w > 1.0 and obstacle_bbox is not None:
                                w_map = self._bbox_weight_map(
                                    obstacle_bbox[:, idx], recon_ss.shape,
                                    rollout_recon_obs_w, self._device,
                                )
                                mse_ss = mse_ss * w_map
                            step_w = (1.0 - rollout_recon_decay) ** (h - 1)
                            rr_loss = rr_loss + step_w * mse_ss.mean().clamp(max=1e4)
                            n_rr_ss += 1
                    if do_rr and n_rr_ss > 0:
                        rr_loss = rr_loss / n_rr_ss
                        loss = loss + eff_rr_weight * rr_loss
                        rr_val = rr_loss.item()
                    if do_cycle and n_cyc > 0:
                        cycle_loss = cycle_loss / n_cyc
                        loss = loss + eff_cycle_weight * cycle_loss
                        cycle_val = cycle_loss.item()

                # ── Per-timestep rollout (default, every t) ──────────────────
                if (not rollout_single_shot) and (do_rr or do_cycle) and t + 1 < T:
                    n_rr = min(rr_horizon_eff, T - 1 - t)
                    if n_rr > 0:
                        state_rr = type(post)(*[x.detach() for x in post])
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
                                # Clamp per-step latent error so an early diverging
                                # prior cannot explode the cycle loss.
                                cycle_loss = cycle_loss + sq.mean().clamp(max=100.0)
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
                                # Clamp per-step recon so an Inf/huge mse from a
                                # diverging rollout latent does not propagate.
                                rr_loss = rr_loss + step_w * mse_rr.mean().clamp(max=100.0)

                        if do_rr:
                            n_rr_used = min(rollout_recon_horizon, T - 1 - t)
                            if n_rr_used > 0:
                                rr_loss = rr_loss / n_rr_used
                                loss    = loss + eff_rr_weight * rr_loss
                                rr_val  = rr_loss.item()  # raw (pre-weight) for monitoring
                        if do_cycle and n_cyc > 0:
                            cycle_loss = cycle_loss / n_cyc
                            loss = loss + eff_cycle_weight * cycle_loss
                            cycle_val = cycle_loss.item()  # raw (pre-weight) for monitoring

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
                    state_d = type(post)(*[x.detach() for x in post])
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

        self._train_step += 1  # advance warmup counters

        # Expose warmup ramps so the trainer can log whether warmup is active.
        return (
            {k: v / T for k, v in accum.items()}
            | {"total_loss": total_loss_value,
               "rr_ramp": rr_ramp, "cycle_ramp": ramp,
               "eff_rr_weight": eff_rr_weight,
               "eff_cycle_weight": eff_cycle_weight}
        )

    def _target_rollout_forward(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_bbox: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Transition-stage target posterior rollout objective."""
        if self.target_world_model is None:
            raise RuntimeError(
                "transition_loss_mode='target_rollout' requires "
                "target_world_model. Pass --init_from with an AE checkpoint."
            )

        wm = self.world_model
        target_wm = self.target_world_model
        cfg = self._cfg.training
        B, T = images.shape[:2]

        context = int(getattr(cfg, "rollout_context_frames", 25)) - 1
        horizon = int(getattr(cfg, "rollout_horizon", 1))
        context = max(0, context)
        horizon = max(1, horizon)
        if context >= T - 1:
            zero = torch.zeros((), device=self._device)
            return zero, {
                "kl_raw": zero,
                "kl_clamped": zero,
                "kl_dyn": zero,
                "rollout_pix_mse": zero,
                "rollout_recon": zero,
                "img_loss": zero,
                "state_loss": zero,
                "kl_rep": zero,
                "obs_aux_loss": zero,
                "cycle": zero,
                "overshoot_kl": zero,
                "kl_train": zero,
                "posterior_anchor_loss": zero,
                "deter_loss": zero,
                "target_img_loss": zero,
                "online_img_loss": zero,
                "target_future_img_loss": zero,
                "online_future_img_loss": zero,
                "rollout_bbox_mse": zero,
                "rollout_plain_pix_mse": zero,
                "bbox_valid_frac": zero,
                "bbox_deter_loss": zero,
                "bbox_token_count": zero,
                "obstacle_valid_count": zero,
                "obstacle_valid_frac": zero,
                "rollout_bbox_mse_valid_only": zero,
                "rollout_plain_pix_mse_valid_only": zero,
                "bbox_deter_loss_valid_only": zero,
                "bbox_token_count_valid_only": zero,
                "bbox_valid_count_raw": zero,
                "bbox_total_count_raw": zero,
                "bbox_valid_frac_raw": zero,
                "gt_future_var": zero,
                "target_recon_future_var": zero,
                "online_recon_future_var": zero,
                "rollout_future_var": zero,
                "rollout_var_ratio": zero,
                "action_sensitivity_left_right_mse": zero,
                "action_sensitivity_gt_left_mse": zero,
                "action_sensitivity_gt_right_mse": zero,
                "lambda_pix_eff": zero,
                "stoch_mean_loss": zero,
                "stoch_decode_loss": zero,
                "lambda_stoch_mean": zero,
                "lambda_stoch_decode": zero,
                "target_rollout_step_decay": zero,
                "target_rollout_future_ramp": zero,
                "h1_weight": zero,
                "h2_weight": zero,
                "total_horizon_weight": zero,
                "total_pixel_weight": zero,
                "target_rollout_detach_between_steps": zero,
                "h1_anchor_loss": zero,
                "h1_anchor_stoch_loss": zero,
                "h1_anchor_deter_loss": zero,
                "h1_anchor_decode_loss": zero,
                "prior_state_denoise_loss": zero,
                "prior_state_denoise_stoch_loss": zero,
                "prior_state_denoise_deter_loss": zero,
                "prior_state_denoise_decode_loss": zero,
            }
        horizon_eff = min(horizon, T - 1 - context)

        target_posts = []
        prev_t = target_wm.rssm.initial(B, self._device)
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", enabled=self._use_amp, dtype=self._amp_dtype,
            ):
                for t in range(T):
                    img_t = target_wm.preprocess_image(images[:, t].to(self._device))
                    st_t = states[:, t].to(self._device)
                    act_t = actions[:, t].to(self._device)
                    embed = target_wm.encoder(img_t, st_t)
                    post_t, _ = target_wm.rssm.obs_step(prev_t, act_t, embed)
                    post_t = self._det_state(post_t)
                    post_t = type(post_t)(*[x.detach() for x in post_t])
                    target_posts.append(post_t)
                    prev_t = post_t

        free_nats_cfg = getattr(cfg, "free_nats_transition", None)
        if free_nats_cfg is None:
            free_nats_cfg = getattr(cfg, "free_nats", 1.0)
        free_nats = float(free_nats_cfg)
        decode_horizons = [
            int(h) for h in getattr(cfg, "rollout_decode_horizons", [horizon_eff])
        ]
        decode_horizons = [
            h for h in decode_horizons if 1 <= h <= horizon_eff
        ] or [horizon_eff]

        kl_raw_sum = torch.zeros((), device=self._device)
        kl_clamped_sum = torch.zeros((), device=self._device)
        deter_sum = torch.zeros((), device=self._device)
        stoch_mean_sum = torch.zeros((), device=self._device)
        stoch_decode_sum = torch.zeros((), device=self._device)
        total_h_weight = torch.zeros((), device=self._device)
        total_pix_weight = torch.zeros((), device=self._device)
        total_bbox_pix_weight = torch.zeros((), device=self._device)
        total_bbox_deter_weight = torch.zeros((), device=self._device)
        pix_sum = torch.zeros((), device=self._device)
        plain_pix_sum = torch.zeros((), device=self._device)
        bbox_pix_sum = torch.zeros((), device=self._device)
        per_horizon_info: dict[str, Tensor] = {}
        bbox_pix_count = 0
        bbox_valid_count = 0
        bbox_total_count = 0
        obstacle_valid_count = 0
        obstacle_total_count = 0
        bbox_plain_valid_sum = torch.zeros((), device=self._device)
        bbox_plain_valid_count = 0
        bbox_deter_valid_sum = torch.zeros((), device=self._device)
        bbox_deter_valid_count = 0
        bbox_token_count_valid_sum = torch.zeros((), device=self._device)
        bbox_deter_sum = torch.zeros((), device=self._device)
        bbox_deter_count = 0
        bbox_token_count_sum = torch.zeros((), device=self._device)
        n_pix = 0
        gt_future_imgs = []
        rollout_future_imgs = []
        bbox_pix_weight = float(getattr(cfg, "bbox_pix_weight", 0.0))
        bbox_deter_weight = float(getattr(cfg, "bbox_deter_weight", 0.0))
        bbox_mask_dilate = int(getattr(cfg, "bbox_mask_dilate", 0))
        bbox_min_area = float(getattr(cfg, "bbox_min_area", 1.0))
        bbox_weight_mode = getattr(cfg, "bbox_weight_mode", "additive")
        bbox_coord_space = getattr(cfg, "bbox_coord_space", "raw")
        _crop = bool(getattr(self._cfg.encoder, "crop_road", False))
        _keep_frac = float(getattr(self._cfg.encoder, "keep_bottom_frac", 0.6))
        _tv = getattr(self._cfg, "token_vit", None)
        anchor_weight = float(getattr(cfg, "posterior_anchor_weight", 0.0))
        if anchor_weight > 0.0:
            anchor_horizons = [
                int(h) for h in getattr(cfg, "posterior_anchor_horizons", [0, 1, 3])
            ]
            anchor_indices = sorted(
                {
                    context + h
                    for h in anchor_horizons
                    if 0 <= context + h < T
                }
            )
        else:
            anchor_indices = []
        max_anchor_idx = max(anchor_indices) if anchor_indices else -1
        online_anchor_posts = {}
        prev_online = wm.rssm.initial(B, self._device)
        decode_deterministic = bool(
            getattr(cfg, "rollout_decode_deterministic", True)
        )
        lambda_stoch_mean = float(getattr(cfg, "lambda_stoch_mean", 0.0))
        lambda_stoch_decode = float(getattr(cfg, "lambda_stoch_decode", 0.0))
        step_decay = float(getattr(cfg, "target_rollout_step_decay", 1.0))
        future_warmup = int(getattr(cfg, "target_rollout_future_warmup_steps", 0))
        detach_between_steps = bool(
            getattr(cfg, "target_rollout_detach_between_steps", False)
        )
        h1_anchor_weight = float(getattr(cfg, "h1_anchor_weight", 0.0))
        h1_anchor_stoch_weight = float(getattr(cfg, "h1_anchor_stoch_weight", 1.0))
        h1_anchor_deter_weight = float(getattr(cfg, "h1_anchor_deter_weight", 1.0))
        h1_anchor_decode_weight = float(getattr(cfg, "h1_anchor_decode_weight", 1.0))
        denoise_weight = float(getattr(cfg, "prior_state_denoise_weight", 0.0))
        denoise_stoch_weight = float(
            getattr(cfg, "prior_state_denoise_stoch_weight", 1.0)
        )
        denoise_deter_weight = float(
            getattr(cfg, "prior_state_denoise_deter_weight", 1.0)
        )
        denoise_decode_weight = float(
            getattr(cfg, "prior_state_denoise_decode_weight", 1.0)
        )
        denoise_start_h = max(
            1, int(getattr(cfg, "prior_state_denoise_start_horizon", 1))
        )
        if h1_anchor_weight > 0.0 and self.h1_anchor_world_model is None:
            raise RuntimeError(
                "training.h1_anchor_weight > 0 requires h1_anchor_world_model. "
                "Set training.h1_anchor_checkpoint to an H1 checkpoint."
            )
        if future_warmup > 0:
            future_ramp = min(1.0, self._train_step / future_warmup)
        else:
            future_ramp = 1.0

        def horizon_weight(h: int) -> Tensor:
            w = step_decay ** max(0, h - 1)
            if h > 1:
                w *= future_ramp
            return torch.tensor(float(w), device=self._device)

        def bbox_for_model(bbox_h: Tensor, image_h: int, image_w: int) -> Tensor:
            return bbox_xyxy_to_model_space(
                bbox_h.to(self._device),
                image_h=image_h,
                image_w=image_w,
                crop_road=_crop,
                keep_bottom_frac=_keep_frac,
                coord_space=bbox_coord_space,
                min_area=bbox_min_area,
            )

        state_roll = type(target_posts[context])(
            *[x.detach() for x in target_posts[context]]
        )
        h1_anchor_loss = torch.zeros((), device=self._device)
        h1_anchor_stoch_loss = torch.zeros((), device=self._device)
        h1_anchor_deter_loss = torch.zeros((), device=self._device)
        h1_anchor_decode_loss = torch.zeros((), device=self._device)
        prior_state_denoise_loss = torch.zeros((), device=self._device)
        prior_state_denoise_stoch_loss = torch.zeros((), device=self._device)
        prior_state_denoise_deter_loss = torch.zeros((), device=self._device)
        prior_state_denoise_decode_loss = torch.zeros((), device=self._device)
        with torch.amp.autocast(
            "cuda", enabled=self._use_amp, dtype=self._amp_dtype,
        ):
            if max_anchor_idx >= 0:
                for t in range(max_anchor_idx + 1):
                    img_t = wm.preprocess_image(images[:, t].to(self._device))
                    st_t = states[:, t].to(self._device)
                    act_t = actions[:, t].to(self._device)
                    embed = wm.encoder(img_t, st_t)
                    post_online, _ = wm.rssm.obs_step(prev_online, act_t, embed)
                    if t in anchor_indices:
                        online_anchor_posts[t] = post_online
                    prev_online = type(post_online)(
                        *[x.detach() for x in post_online]
                    )

            for h in range(1, horizon_eff + 1):
                idx = context + h
                weight_h = horizon_weight(h)
                total_h_weight = total_h_weight + weight_h
                act_h = actions[:, idx].to(self._device)
                state_roll = wm.rssm.img_step(state_roll, act_h)

                prior_mean, prior_std = wm.get_kl_stats(state_roll)
                target_mean, target_std = target_wm.get_kl_stats(target_posts[idx])
                kl_raw, kl_clamped = compute_transition_target_kl(
                    prior_mean,
                    prior_std,
                    target_mean,
                    target_std,
                    free_nats=free_nats,
                )
                kl_raw_sum = kl_raw_sum + weight_h * kl_raw
                kl_clamped_sum = kl_clamped_sum + weight_h * kl_clamped
                deter_sq = (state_roll.deter - target_posts[idx].deter.detach()).pow(2)
                plain_deter = deter_sq.mean()
                deter_h = plain_deter
                stoch_mean_h = self._stoch_mean_alignment_loss(
                    state_roll, target_posts[idx],
                )
                per_horizon_info[f"kl_raw_h{h}"] = kl_raw.detach()
                per_horizon_info[f"stoch_mean_loss_h{h}"] = stoch_mean_h.detach()
                if lambda_stoch_mean > 0.0:
                    stoch_mean_sum = stoch_mean_sum + weight_h * stoch_mean_h

                target_img_step = wm.preprocess_image(images[:, idx].to(self._device))
                if lambda_stoch_decode > 0.0:
                    hybrid_state = self._hybrid_prior_stoch_state(
                        target_posts[idx], state_roll,
                    )
                    hybrid_img = wm.decode_obs(hybrid_state)
                    stoch_decode_h = (
                        hybrid_img - target_img_step
                    ).pow(2).mean()
                    stoch_decode_sum = stoch_decode_sum + weight_h * stoch_decode_h
                else:
                    with torch.no_grad():
                        hybrid_state = self._hybrid_prior_stoch_state(
                            target_posts[idx], state_roll,
                        )
                        hybrid_img = wm.decode_obs(hybrid_state)
                        stoch_decode_h = (
                            hybrid_img - target_img_step
                        ).pow(2).mean()
                per_horizon_info[f"stoch_decode_loss_h{h}"] = stoch_decode_h.detach()

                if h == 1 and h1_anchor_weight > 0.0:
                    anchor_wm = self.h1_anchor_world_model
                    assert anchor_wm is not None
                    with torch.no_grad():
                        anchor_prev = self._detach_state(target_posts[context])
                        anchor_h1 = anchor_wm.rssm.img_step(anchor_prev, act_h)
                        anchor_h1 = self._detach_state(anchor_h1)
                        anchor_img = anchor_wm.decode_obs(self._det_state(anchor_h1))
                    h1_anchor_stoch_loss = self._stoch_mean_alignment_loss(
                        state_roll, anchor_h1,
                    )
                    h1_anchor_deter_loss = nn.functional.mse_loss(
                        state_roll.deter, anchor_h1.deter.detach(),
                    )
                    current_anchor_img = wm.decode_obs(self._det_state(state_roll))
                    h1_anchor_decode_loss = nn.functional.mse_loss(
                        current_anchor_img, anchor_img.detach(),
                    )
                    h1_anchor_loss = (
                        h1_anchor_stoch_weight * h1_anchor_stoch_loss
                        + h1_anchor_deter_weight * h1_anchor_deter_loss
                        + h1_anchor_decode_weight * h1_anchor_decode_loss
                    )

                if (
                    obstacle_bbox is not None
                    and hasattr(state_roll, "token_mean")
                    and _tv is not None
                ):
                    bbox_h = bbox_for_model(
                        obstacle_bbox[:, idx], _tv.image_size, _tv.image_size,
                    )
                    tok_mask = bbox_xyxy_to_token_mask(
                        bbox_h,
                        image_h=_tv.image_size,
                        image_w=_tv.image_size,
                        patch_size=_tv.patch_size,
                        min_area=bbox_min_area,
                    ).to(self._device).float()
                    if tok_mask.shape[1] != state_roll.deter.shape[1]:
                        tok_mask = torch.zeros(
                            B, state_roll.deter.shape[1], device=self._device,
                        )
                    tok_count = tok_mask.sum()
                    bbox_token_count_sum = bbox_token_count_sum + tok_count.detach()
                    if tok_count > 0:
                        bbox_deter = (
                            deter_sq.mean(dim=-1) * tok_mask
                        ).sum() / tok_count
                        if bbox_deter_weight > 0.0:
                            deter_h = plain_deter + bbox_deter_weight * bbox_deter
                        bbox_deter_sum = bbox_deter_sum + weight_h * bbox_deter.detach()
                        total_bbox_deter_weight = total_bbox_deter_weight + weight_h
                        bbox_deter_count += 1
                        if h in decode_horizons:
                            bbox_deter_valid_sum = (
                                bbox_deter_valid_sum + weight_h * bbox_deter.detach()
                            )
                            bbox_deter_valid_count += 1
                            bbox_token_count_valid_sum = (
                                bbox_token_count_valid_sum + tok_count.detach()
                            )
                per_horizon_info[f"deter_loss_h{h}"] = deter_h.detach()
                deter_sum = deter_sum + weight_h * deter_h

                if h in decode_horizons:
                    total_pix_weight = total_pix_weight + weight_h
                    # Decoder parameters are frozen, but this forward must keep
                    # gradients to the rollout latent/RSSM prior.
                    decode_state = (
                        self._det_state(state_roll)
                        if decode_deterministic else state_roll
                    )
                    recon_h = wm.decode_obs(decode_state)
                    target_img = target_img_step
                    gt_future_imgs.append(target_img.detach())
                    rollout_future_imgs.append(recon_h.detach())
                    pix_err = (recon_h - target_img).pow(2)
                    plain_mse = pix_err.mean()
                    per_horizon_info[f"rollout_pix_mse_h{h}"] = plain_mse.detach()
                    per_horizon_info[f"rollout_plain_pix_mse_h{h}"] = plain_mse.detach()
                    pix_h = plain_mse
                    plain_pix_sum = plain_pix_sum + weight_h * plain_mse.detach()
                    if obstacle_bbox is not None:
                        _, _, img_h, img_w = pix_err.shape
                        bbox_h = bbox_for_model(obstacle_bbox[:, idx], img_h, img_w)
                        mask_h, valid_h = bbox_xyxy_to_mask(
                            bbox_h,
                            image_h=img_h,
                            image_w=img_w,
                            dilate=bbox_mask_dilate,
                            min_area=bbox_min_area,
                        )
                        mask_h = mask_h.to(self._device)
                        valid_h = valid_h.to(self._device)
                        bbox_total_count += B
                        bbox_valid_count += int(valid_h.sum().item())
                        obstacle_total_count += B
                        obstacle_valid_count += int(valid_h.sum().item())
                        if valid_h.any():
                            bbox_plain_valid_sum = (
                                bbox_plain_valid_sum
                                + pix_err.mean(dim=(1, 2, 3))[valid_h].sum().detach()
                            )
                            bbox_plain_valid_count += int(valid_h.sum().item())
                        denom = mask_h.sum() * pix_err.shape[1]
                        if denom > 0:
                            bbox_mse = (pix_err * mask_h).sum() / denom
                            bbox_pix_sum = bbox_pix_sum + weight_h * bbox_mse.detach()
                            total_bbox_pix_weight = total_bbox_pix_weight + weight_h
                            bbox_pix_count += 1
                            if bbox_pix_weight > 0.0 and bbox_weight_mode == "weighted_mean":
                                weights = 1.0 + bbox_pix_weight * mask_h
                                pix_h = (pix_err * weights).sum() / (
                                    weights.sum() * pix_err.shape[1]
                                )
                            elif bbox_pix_weight > 0.0:
                                pix_h = plain_mse + bbox_pix_weight * bbox_mse
                    pix_sum = pix_sum + weight_h * pix_h
                    n_pix += 1

                if detach_between_steps and h < horizon_eff:
                    state_roll = self._detach_state(state_roll)

            if denoise_weight > 0.0 and horizon_eff >= denoise_start_h + 1:
                denoise_state = type(target_posts[context])(
                    *[x.detach() for x in target_posts[context]]
                )
                for dh in range(1, denoise_start_h + 1):
                    denoise_idx = context + dh
                    denoise_state = wm.rssm.img_step(
                        denoise_state, actions[:, denoise_idx].to(self._device)
                    )
                denoise_input = self._detach_state(self._det_state(denoise_state))
                denoise_target_h = denoise_start_h + 1
                denoise_target_idx = context + denoise_target_h
                denoise_prior = wm.rssm.img_step(
                    denoise_input,
                    actions[:, denoise_target_idx].to(self._device),
                )
                denoise_target = target_posts[denoise_target_idx]
                prior_state_denoise_stoch_loss = self._stoch_mean_alignment_loss(
                    denoise_prior, denoise_target,
                )
                prior_state_denoise_deter_loss = nn.functional.mse_loss(
                    denoise_prior.deter, denoise_target.deter.detach(),
                )
                denoise_img = wm.decode_obs(self._det_state(denoise_prior))
                denoise_target_img = wm.preprocess_image(
                    images[:, denoise_target_idx].to(self._device)
                )
                prior_state_denoise_decode_loss = nn.functional.mse_loss(
                    denoise_img, denoise_target_img,
                )
                prior_state_denoise_loss = (
                    denoise_stoch_weight * prior_state_denoise_stoch_loss
                    + denoise_deter_weight * prior_state_denoise_deter_loss
                    + denoise_decode_weight * prior_state_denoise_decode_loss
                )

            h_denom = total_h_weight.clamp_min(1e-8)
            pix_denom = total_pix_weight.clamp_min(1e-8)
            bbox_pix_denom = total_bbox_pix_weight.clamp_min(1e-8)
            bbox_deter_denom = total_bbox_deter_weight.clamp_min(1e-8)
            kl_raw_mean = kl_raw_sum / h_denom
            kl_clamped_mean = kl_clamped_sum / h_denom
            deter_loss = deter_sum / h_denom
            stoch_mean_loss = stoch_mean_sum / h_denom
            stoch_decode_loss = stoch_decode_sum / h_denom
            rollout_pix_mse = pix_sum / pix_denom
            rollout_plain_pix_mse = plain_pix_sum / pix_denom
            rollout_bbox_mse = bbox_pix_sum / bbox_pix_denom
            bbox_valid_frac = torch.tensor(
                (bbox_valid_count / bbox_total_count) if bbox_total_count > 0 else 0.0,
                device=self._device,
            )
            bbox_valid_count_raw = torch.tensor(
                float(bbox_valid_count), device=self._device,
            )
            bbox_total_count_raw = torch.tensor(
                float(bbox_total_count), device=self._device,
            )
            bbox_valid_frac_raw = bbox_valid_frac
            bbox_deter_loss = bbox_deter_sum / bbox_deter_denom
            bbox_token_count = bbox_token_count_sum / max(1, horizon_eff)
            obstacle_valid_count_t = torch.tensor(
                float(obstacle_valid_count), device=self._device,
            )
            obstacle_valid_frac = torch.tensor(
                (obstacle_valid_count / obstacle_total_count)
                if obstacle_total_count > 0 else 0.0,
                device=self._device,
            )
            rollout_bbox_mse_valid_only = bbox_pix_sum / max(1, bbox_pix_count)
            rollout_plain_pix_mse_valid_only = (
                bbox_plain_valid_sum / max(1, bbox_plain_valid_count)
            )
            bbox_deter_loss_valid_only = (
                bbox_deter_valid_sum / bbox_deter_denom
            )
            bbox_token_count_valid_only = (
                bbox_token_count_valid_sum / max(1, bbox_deter_valid_count)
            )
            use_raw_kl = bool(getattr(cfg, "transition_use_raw_kl_loss", True))
            kl_train = kl_raw_mean if use_raw_kl else kl_clamped_mean

            posterior_anchor_loss = torch.zeros((), device=self._device)
            n_anchor = 0
            if anchor_weight > 0.0 and online_anchor_posts:
                for idx, online_post in online_anchor_posts.items():
                    target_post = target_posts[idx]
                    if hasattr(online_post, "token_mean"):
                        posterior_anchor_loss = posterior_anchor_loss + (
                            (online_post.token_mean - target_post.token_mean.detach())
                            .pow(2)
                            .mean()
                            + (online_post.deter - target_post.deter.detach())
                            .pow(2)
                            .mean()
                        )
                    else:
                        online_mean, online_std = wm.get_kl_stats(online_post)
                        target_mean, target_std = target_wm.get_kl_stats(target_post)
                        anchor_raw, _ = compute_transition_target_kl(
                            online_mean,
                            online_std,
                            target_mean,
                            target_std,
                            free_nats=0.0,
                        )
                        posterior_anchor_loss = posterior_anchor_loss + anchor_raw
                    n_anchor += 1
                posterior_anchor_loss = posterior_anchor_loss / max(1, n_anchor)

            lambda_kl = float(getattr(cfg, "lambda_kl", 1.0))
            lambda_pix = float(getattr(cfg, "lambda_pix", 0.0))
            lambda_deter = float(getattr(cfg, "lambda_deter", 1.0))
            pix_warmup = int(getattr(cfg, "rollout_pix_warmup_steps", 0))
            if pix_warmup > 0:
                pix_ramp = min(1.0, self._train_step / pix_warmup)
            else:
                pix_ramp = 1.0
            lambda_pix_eff = lambda_pix * pix_ramp
            loss = (
                lambda_kl * kl_train
                + lambda_deter * deter_loss
                + lambda_pix_eff * rollout_pix_mse
                + anchor_weight * posterior_anchor_loss
                + lambda_stoch_mean * stoch_mean_loss
                + lambda_stoch_decode * stoch_decode_loss
                + h1_anchor_weight * h1_anchor_loss
                + denoise_weight * prior_state_denoise_loss
            )

        with torch.no_grad():
            ctx = context
            img_ctx = wm.preprocess_image(images[:, ctx].to(self._device))
            st_ctx = states[:, ctx].to(self._device)
            target_det_ctx = self._det_state(target_posts[ctx])
            target_recon_ctx = wm.decode_obs(target_det_ctx)
            target_feat_ctx = wm.rssm.get_feat(target_det_ctx)
            target_recon_state_ctx = wm.state_decoder(target_feat_ctx)
            target_img_loss = (target_recon_ctx - img_ctx).pow(2).mean()

            online_ctx = online_anchor_posts.get(ctx)
            if online_ctx is None:
                prev_metric = wm.rssm.initial(B, self._device)
                for t in range(ctx + 1):
                    img_t = wm.preprocess_image(images[:, t].to(self._device))
                    st_t = states[:, t].to(self._device)
                    act_t = actions[:, t].to(self._device)
                    emb_t = wm.encoder(img_t, st_t)
                    online_ctx, _ = wm.rssm.obs_step(prev_metric, act_t, emb_t)
                    prev_metric = type(online_ctx)(
                        *[x.detach() for x in online_ctx]
                    )
            online_det_ctx = self._det_state(online_ctx)
            online_recon_ctx = wm.decode_obs(online_det_ctx)
            online_feat_ctx = wm.rssm.get_feat(online_det_ctx)
            online_recon_state_ctx = wm.state_decoder(online_feat_ctx)
            online_img_loss = (online_recon_ctx - img_ctx).pow(2).mean()

            target_future_sum = torch.zeros((), device=self._device)
            online_future_sum = torch.zeros((), device=self._device)
            n_future = 0
            target_future_imgs = []
            online_future_imgs = []
            gt_metric_imgs = []
            future_indices = {
                context + h
                for h in decode_horizons
                if 0 <= context + h < T
            }
            online_future_posts = {}
            if future_indices:
                prev_future = wm.rssm.initial(B, self._device)
                for t in range(max(future_indices) + 1):
                    img_t = wm.preprocess_image(images[:, t].to(self._device))
                    st_t = states[:, t].to(self._device)
                    act_t = actions[:, t].to(self._device)
                    emb_t = wm.encoder(img_t, st_t)
                    online_post_t, _ = wm.rssm.obs_step(prev_future, act_t, emb_t)
                    if t in future_indices:
                        online_future_posts[t] = online_post_t
                    prev_future = type(online_post_t)(
                        *[x.detach() for x in online_post_t]
                    )

            for idx in sorted(future_indices):
                img_future = wm.preprocess_image(images[:, idx].to(self._device))
                target_future = wm.decode_obs(self._det_state(target_posts[idx]))
                online_future = wm.decode_obs(
                    self._det_state(online_future_posts[idx])
                )
                gt_metric_imgs.append(img_future)
                target_future_imgs.append(target_future)
                online_future_imgs.append(online_future)
                target_future_sum = target_future_sum + (
                    target_future - img_future
                ).pow(2).mean()
                online_future_sum = online_future_sum + (
                    online_future - img_future
                ).pow(2).mean()
                n_future += 1
            target_future_img_loss = target_future_sum / max(1, n_future)
            online_future_img_loss = online_future_sum / max(1, n_future)

            def _stack_var(xs):
                if not xs:
                    return torch.zeros((), device=self._device)
                return torch.stack(xs, dim=0).float().var(dim=1, unbiased=False).mean()

            gt_future_var = _stack_var(gt_metric_imgs)
            target_recon_future_var = _stack_var(target_future_imgs)
            online_recon_future_var = _stack_var(online_future_imgs)
            rollout_future_var = _stack_var(rollout_future_imgs)
            rollout_var_ratio = rollout_future_var / (gt_future_var + 1e-8)

            def _rollout_decode(action_fn):
                outs = []
                s = self._det_state(target_posts[context])
                for h in range(1, horizon_eff + 1):
                    act_h = action_fn(h).to(self._device)
                    s = self._det_state(wm.rssm.img_step(s, act_h))
                    if h in decode_horizons:
                        outs.append(wm.decode_obs(s))
                return outs

            action_dim = actions.shape[-1]
            base_action = actions[:, context].to(self._device).clone()
            left_action = base_action.clone()
            right_action = base_action.clone()
            if action_dim > 0:
                left_action[:, 0] = 0.2
                right_action[:, 0] = -0.2

            gt_action_rollout = _rollout_decode(
                lambda h: actions[:, context + h].to(self._device)
            )
            left_rollout = _rollout_decode(lambda h: left_action)
            right_rollout = _rollout_decode(lambda h: right_action)

            def _seq_mse(a, b):
                if not a or not b:
                    return torch.zeros((), device=self._device)
                return torch.stack(
                    [(x - y).pow(2).mean() for x, y in zip(a, b)]
                ).mean()

            action_sensitivity_left_right_mse = _seq_mse(left_rollout, right_rollout)
            action_sensitivity_gt_left_mse = _seq_mse(gt_action_rollout, left_rollout)
            action_sensitivity_gt_right_mse = _seq_mse(gt_action_rollout, right_rollout)

            img_loss = online_img_loss
            from active_inference.utils.transforms import symlog
            state_loss = nn.functional.mse_loss(
                symlog(online_recon_state_ctx), symlog(st_ctx),
            )

        zero = torch.zeros((), device=self._device)
        info = {
            "kl_raw": kl_raw_mean,
            "kl_clamped": kl_clamped_mean,
            "kl_dyn": kl_clamped_mean,
            "kl_train": kl_train,
            "deter_loss": deter_loss,
            "rollout_pix_mse": rollout_pix_mse,
            "rollout_recon": rollout_pix_mse,
            "img_loss": img_loss,
            "state_loss": state_loss,
            "target_img_loss": target_img_loss,
            "online_img_loss": online_img_loss,
            "target_future_img_loss": target_future_img_loss,
            "online_future_img_loss": online_future_img_loss,
            "rollout_bbox_mse": rollout_bbox_mse,
            "rollout_plain_pix_mse": rollout_plain_pix_mse,
            "bbox_valid_frac": bbox_valid_frac,
            "bbox_deter_loss": bbox_deter_loss,
            "bbox_token_count": bbox_token_count,
            "obstacle_valid_count": obstacle_valid_count_t,
            "obstacle_valid_frac": obstacle_valid_frac,
            "rollout_bbox_mse_valid_only": rollout_bbox_mse_valid_only,
            "rollout_plain_pix_mse_valid_only": rollout_plain_pix_mse_valid_only,
            "bbox_deter_loss_valid_only": bbox_deter_loss_valid_only,
            "bbox_token_count_valid_only": bbox_token_count_valid_only,
            "bbox_valid_count_raw": bbox_valid_count_raw,
            "bbox_total_count_raw": bbox_total_count_raw,
            "bbox_valid_frac_raw": bbox_valid_frac_raw,
            "gt_future_var": gt_future_var,
            "target_recon_future_var": target_recon_future_var,
            "online_recon_future_var": online_recon_future_var,
            "rollout_future_var": rollout_future_var,
            "rollout_var_ratio": rollout_var_ratio,
            "action_sensitivity_left_right_mse": action_sensitivity_left_right_mse,
            "action_sensitivity_gt_left_mse": action_sensitivity_gt_left_mse,
            "action_sensitivity_gt_right_mse": action_sensitivity_gt_right_mse,
            "kl_rep": zero,
            "obs_aux_loss": zero,
            "cycle": zero,
            "overshoot_kl": zero,
            "posterior_anchor_loss": posterior_anchor_loss,
            "lambda_pix_eff": torch.tensor(lambda_pix_eff, device=self._device),
            "stoch_mean_loss": stoch_mean_loss,
            "stoch_decode_loss": stoch_decode_loss,
            "lambda_stoch_mean": torch.tensor(lambda_stoch_mean, device=self._device),
            "lambda_stoch_decode": torch.tensor(lambda_stoch_decode, device=self._device),
            "target_rollout_step_decay": torch.tensor(step_decay, device=self._device),
            "target_rollout_future_ramp": torch.tensor(future_ramp, device=self._device),
            "h1_weight": horizon_weight(1),
            "h2_weight": horizon_weight(2) if horizon_eff >= 2 else zero,
            "total_horizon_weight": total_h_weight,
            "total_pixel_weight": total_pix_weight,
            "target_rollout_detach_between_steps": torch.tensor(
                1.0 if detach_between_steps else 0.0, device=self._device,
            ),
            "h1_anchor_loss": h1_anchor_loss,
            "h1_anchor_stoch_loss": h1_anchor_stoch_loss,
            "h1_anchor_deter_loss": h1_anchor_deter_loss,
            "h1_anchor_decode_loss": h1_anchor_decode_loss,
            "prior_state_denoise_loss": prior_state_denoise_loss,
            "prior_state_denoise_stoch_loss": prior_state_denoise_stoch_loss,
            "prior_state_denoise_deter_loss": prior_state_denoise_deter_loss,
            "prior_state_denoise_decode_loss": prior_state_denoise_decode_loss,
        }
        info.update(per_horizon_info)
        return loss, info

    def update_transition_target_rollout(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_bbox: Tensor | None = None,
    ) -> dict[str, float]:
        self._optimizer.zero_grad()
        loss, info_t = self._target_rollout_forward(
            images, states, actions, obstacle_bbox=obstacle_bbox,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            return {k: float(v.detach().item()) for k, v in info_t.items()} | {
                "total_loss": float("nan")
            }

        if loss.requires_grad:
            loss.backward()
            clip_params = self._optimizer.param_groups[0]["params"]
            nn.utils.clip_grad_norm_(clip_params, self._cfg.training.grad_clip)
            self._optimizer.step()

        sched = getattr(self, "_scheduler", None)
        if sched is not None:
            sched.step()
        self._train_step += 1

        return {k: float(v.detach().item()) for k, v in info_t.items()} | {
            "total_loss": float(loss.detach().item())
        }

    @torch.no_grad()
    def evaluate_transition_target_rollout_batch(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_bbox: Tensor | None = None,
    ) -> dict[str, float]:
        loss, info_t = self._target_rollout_forward(
            images, states, actions, obstacle_bbox=obstacle_bbox,
        )
        return {k: float(v.detach().item()) for k, v in info_t.items()} | {
            "total_loss": float(loss.detach().item())
        }

    def _dense_one_step_forward(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Dense teacher-forced one-step prior objective for transition stage."""
        cfg = self._cfg.training
        if self._stage != "transition":
            raise RuntimeError(
                "transition_loss_mode='dense_one_step' requires stage='transition'."
            )
        if not bool(getattr(cfg, "transition_use_target_model", False)):
            raise RuntimeError(
                "transition_loss_mode='dense_one_step' requires "
                "training.transition_use_target_model=true."
            )
        if self.target_world_model is None:
            raise RuntimeError(
                "transition_loss_mode='dense_one_step' requires "
                "target_world_model. Pass --init_from with an AE checkpoint."
            )

        wm = self.world_model
        target_wm = self.target_world_model
        B, T = images.shape[:2]
        zero = torch.zeros((), device=self._device)
        if T < 2:
            return zero, self._dense_one_step_info_defaults(zero)

        target_posts = []
        prev_t = target_wm.rssm.initial(B, self._device)
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", enabled=self._use_amp, dtype=self._amp_dtype,
            ):
                for t in range(T):
                    img_t = target_wm.preprocess_image(images[:, t].to(self._device))
                    st_t = states[:, t].to(self._device)
                    act_t = actions[:, t].to(self._device)
                    embed = target_wm.encoder(img_t, st_t)
                    post_t, _ = target_wm.rssm.obs_step(prev_t, act_t, embed)
                    post_t = self._det_state(post_t)
                    post_t = type(post_t)(*[x.detach() for x in post_t])
                    target_posts.append(post_t)
                    prev_t = post_t

        free_nats_cfg = getattr(cfg, "free_nats_transition", None)
        if free_nats_cfg is None:
            free_nats_cfg = getattr(cfg, "free_nats", 1.0)
        free_nats = float(free_nats_cfg)
        decode_deterministic = bool(
            getattr(cfg, "rollout_decode_deterministic", True)
        )

        kl_raw_sum = torch.zeros((), device=self._device)
        kl_clamped_sum = torch.zeros((), device=self._device)
        deter_sum = torch.zeros((), device=self._device)
        stoch_mean_sum = torch.zeros((), device=self._device)
        stoch_decode_sum = torch.zeros((), device=self._device)
        pix_sum = torch.zeros((), device=self._device)
        gt_imgs = []
        rollout_imgs = []
        lambda_stoch_mean = float(getattr(cfg, "lambda_stoch_mean", 0.0))
        lambda_stoch_decode = float(getattr(cfg, "lambda_stoch_decode", 0.0))
        with torch.amp.autocast(
            "cuda", enabled=self._use_amp, dtype=self._amp_dtype,
        ):
            for t in range(1, T):
                state_prev = type(target_posts[t - 1])(
                    *[x.detach() for x in target_posts[t - 1]]
                )
                target_t = target_posts[t]
                prior_t = wm.rssm.img_step(
                    state_prev, actions[:, t].to(self._device)
                )

                prior_mean, prior_std = wm.get_kl_stats(prior_t)
                target_mean, target_std = target_wm.get_kl_stats(target_t)
                kl_raw, kl_clamped = compute_transition_target_kl(
                    prior_mean,
                    prior_std,
                    target_mean,
                    target_std,
                    free_nats=free_nats,
                )
                kl_raw_sum = kl_raw_sum + kl_raw
                kl_clamped_sum = kl_clamped_sum + kl_clamped
                deter_sum = deter_sum + (
                    prior_t.deter - target_t.deter.detach()
                ).pow(2).mean()
                if lambda_stoch_mean > 0.0:
                    stoch_mean_sum = stoch_mean_sum + self._stoch_mean_alignment_loss(
                        prior_t, target_t,
                    )

                decode_state = (
                    self._det_state(prior_t) if decode_deterministic else prior_t
                )
                recon_t = wm.decode_obs(decode_state)
                gt_t = wm.preprocess_image(images[:, t].to(self._device))
                pix_sum = pix_sum + (recon_t - gt_t).pow(2).mean()
                if lambda_stoch_decode > 0.0:
                    hybrid_state = self._hybrid_prior_stoch_state(target_t, prior_t)
                    hybrid_img = wm.decode_obs(hybrid_state)
                    stoch_decode_sum = stoch_decode_sum + (
                        hybrid_img - gt_t
                    ).pow(2).mean()
                gt_imgs.append(gt_t.detach())
                rollout_imgs.append(recon_t.detach())

            denom = max(1, T - 1)
            kl_raw_mean = kl_raw_sum / denom
            kl_clamped_mean = kl_clamped_sum / denom
            deter_loss = deter_sum / denom
            stoch_mean_loss = stoch_mean_sum / denom
            stoch_decode_loss = stoch_decode_sum / denom
            rollout_pix_mse = pix_sum / denom
            use_raw_kl = bool(getattr(cfg, "transition_use_raw_kl_loss", True))
            kl_train = kl_raw_mean if use_raw_kl else kl_clamped_mean

            lambda_kl = float(getattr(cfg, "lambda_kl", 1.0))
            lambda_deter = float(getattr(cfg, "lambda_deter", 1.0))
            lambda_pix_eff = float(getattr(cfg, "lambda_pix", 0.0))
            loss = (
                lambda_kl * kl_train
                + lambda_deter * deter_loss
                + lambda_pix_eff * rollout_pix_mse
                + lambda_stoch_mean * stoch_mean_loss
                + lambda_stoch_decode * stoch_decode_loss
            )

        def _seq_var(xs):
            if not xs:
                return torch.zeros((), device=self._device)
            return torch.cat(xs, dim=0).float().var(dim=0, unbiased=False).mean()

        gt_future_var = _seq_var(gt_imgs)
        rollout_future_var = _seq_var(rollout_imgs)
        rollout_var_ratio = rollout_future_var / (gt_future_var + 1e-8)

        info = self._dense_one_step_info_defaults(zero)
        info.update({
            "kl_raw": kl_raw_mean,
            "kl_clamped": kl_clamped_mean,
            "kl_dyn": kl_clamped_mean,
            "kl_train": kl_train,
            "deter_loss": deter_loss,
            "rollout_pix_mse": rollout_pix_mse,
            "rollout_recon": rollout_pix_mse,
            "rollout_plain_pix_mse": rollout_pix_mse,
            "img_loss": rollout_pix_mse,
            "gt_future_var": gt_future_var,
            "rollout_future_var": rollout_future_var,
            "rollout_var_ratio": rollout_var_ratio,
            "lambda_pix_eff": torch.tensor(lambda_pix_eff, device=self._device),
            "stoch_mean_loss": stoch_mean_loss,
            "stoch_decode_loss": stoch_decode_loss,
            "lambda_stoch_mean": torch.tensor(lambda_stoch_mean, device=self._device),
            "lambda_stoch_decode": torch.tensor(lambda_stoch_decode, device=self._device),
        })
        return loss, info

    def _dense_one_step_info_defaults(self, zero: Tensor) -> dict[str, Tensor]:
        return {
            "kl_raw": zero,
            "kl_clamped": zero,
            "kl_dyn": zero,
            "kl_train": zero,
            "deter_loss": zero,
            "rollout_pix_mse": zero,
            "rollout_recon": zero,
            "rollout_plain_pix_mse": zero,
            "img_loss": zero,
            "state_loss": zero,
            "kl_rep": zero,
            "obs_aux_loss": zero,
            "cycle": zero,
            "overshoot_kl": zero,
            "posterior_anchor_loss": zero,
            "target_img_loss": zero,
            "online_img_loss": zero,
            "target_future_img_loss": zero,
            "online_future_img_loss": zero,
            "rollout_bbox_mse": zero,
            "bbox_valid_frac": zero,
            "bbox_deter_loss": zero,
            "bbox_token_count": zero,
            "obstacle_valid_count": zero,
            "obstacle_valid_frac": zero,
            "rollout_bbox_mse_valid_only": zero,
            "rollout_plain_pix_mse_valid_only": zero,
            "bbox_deter_loss_valid_only": zero,
            "bbox_token_count_valid_only": zero,
            "bbox_valid_count_raw": zero,
            "bbox_total_count_raw": zero,
            "bbox_valid_frac_raw": zero,
            "gt_future_var": zero,
            "target_recon_future_var": zero,
            "online_recon_future_var": zero,
            "rollout_future_var": zero,
            "rollout_var_ratio": zero,
            "action_sensitivity_left_right_mse": zero,
            "action_sensitivity_gt_left_mse": zero,
            "action_sensitivity_gt_right_mse": zero,
            "lambda_pix_eff": zero,
            "stoch_mean_loss": zero,
            "stoch_decode_loss": zero,
            "lambda_stoch_mean": zero,
            "lambda_stoch_decode": zero,
            "target_rollout_step_decay": zero,
            "target_rollout_future_ramp": zero,
            "h1_weight": zero,
            "h2_weight": zero,
            "total_horizon_weight": zero,
            "total_pixel_weight": zero,
            "target_rollout_detach_between_steps": zero,
            "h1_anchor_loss": zero,
            "h1_anchor_stoch_loss": zero,
            "h1_anchor_deter_loss": zero,
            "h1_anchor_decode_loss": zero,
            "prior_state_denoise_loss": zero,
            "prior_state_denoise_stoch_loss": zero,
            "prior_state_denoise_deter_loss": zero,
            "prior_state_denoise_decode_loss": zero,
        }

    def update_transition_dense_one_step(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
    ) -> dict[str, float]:
        self._optimizer.zero_grad()
        loss, info_t = self._dense_one_step_forward(images, states, actions)

        if torch.isnan(loss) or torch.isinf(loss):
            return {k: float(v.detach().item()) for k, v in info_t.items()} | {
                "total_loss": float("nan")
            }

        if loss.requires_grad:
            loss.backward()
            clip_params = self._optimizer.param_groups[0]["params"]
            nn.utils.clip_grad_norm_(clip_params, self._cfg.training.grad_clip)
            self._optimizer.step()

        sched = getattr(self, "_scheduler", None)
        if sched is not None:
            sched.step()
        self._train_step += 1

        return {k: float(v.detach().item()) for k, v in info_t.items()} | {
            "total_loss": float(loss.detach().item())
        }

    @torch.no_grad()
    def evaluate_transition_dense_one_step_batch(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
    ) -> dict[str, float]:
        loss, info_t = self._dense_one_step_forward(images, states, actions)
        return {k: float(v.detach().item()) for k, v in info_t.items()} | {
            "total_loss": float(loss.detach().item())
        }

    def _stoch_mean_alignment_loss(self, prior, target) -> Tensor:
        if hasattr(prior, "token_mean") and hasattr(target, "token_mean"):
            return nn.functional.mse_loss(
                prior.token_mean, target.token_mean.detach(),
            )
        return nn.functional.mse_loss(prior.mean, target.mean.detach())

    def _hybrid_prior_stoch_state(self, target, prior):
        if hasattr(prior, "token_mean"):
            return type(prior)(
                deter=target.deter.detach(),
                stoch=prior.token_mean,
                mean=prior.mean,
                std=prior.std,
                token_mean=prior.token_mean,
                token_std=prior.token_std,
            )
        return type(prior)(
            deter=target.deter.detach(),
            stoch=prior.mean,
            mean=prior.mean,
            std=prior.std,
        )

    @staticmethod
    def _detach_state(state):
        return type(state)(*[x.detach() for x in state])

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
        untouched. Kept for backward compatibility; transition warm-starts
        should prefer load_world_model_weights().
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

    def load_world_model_weights(self, path: str | Path):
        """Load full world_model weights without optimizer or preference state."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self.world_model.load_state_dict(ckpt["world_model"])
        print(f"  Loaded full world_model weights from {path}")

    def set_target_world_model_from_checkpoint(self, path: str | Path):
        """Create a frozen target world model from a checkpoint."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        target = WorldModel(self._cfg).to(self._device)
        target.load_state_dict(ckpt["world_model"])
        target.eval()
        for p in target.parameters():
            p.requires_grad = False
        self.target_world_model = target
        print(f"  Loaded frozen target_world_model from {path}")

    def set_h1_anchor_world_model_from_checkpoint(self, path: str | Path):
        """Create a frozen H1 anchor world model from a checkpoint."""
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        anchor = WorldModel(self._cfg).to(self._device)
        anchor.load_state_dict(ckpt["world_model"])
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad = False
        self.h1_anchor_world_model = anchor
        print(f"  Loaded frozen h1_anchor_world_model from {path}")

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
