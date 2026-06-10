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

        self._optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.training.lr)
        self._use_amp = "cuda" in str(self._device)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None
        self._prev_state: RSSMState | None = None
        self._prev_action: Tensor | None = None

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

    def update(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        obstacle_labels: Tensor | None = None,
    ) -> dict[str, float]:
        # Per-timestep backward with NaN guard, AMP, and CUDA error recovery
        # obstacle_labels: [B, T] binary (1=obstacle visible, 0=clear)
        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
        cfg = self._cfg.training
        beta_obstacle_aux = getattr(cfg, "beta_obstacle_aux", 0.0)

        overshoot_horizon = getattr(cfg, "overshoot_horizon", 0)
        overshoot_weight = getattr(cfg, "overshoot_weight", 0.0)
        token_kl_weighting = getattr(cfg, "token_kl_weighting", "none")
        warp_smoothness_weight = getattr(cfg, "warp_smoothness_weight", 0.0)

        self._optimizer.zero_grad()
        total_loss_value = 0.0
        accum = {
            "img_loss": 0.0, "state_loss": 0.0,
            "kl_dyn": 0.0, "kl_rep": 0.0, "obs_aux_loss": 0.0,
            "overshoot_kl": 0.0,
        }

        # ── 1st pass: collect stop-grad posterior references for overshooting ──
        # Runs encoder + obs_step once under no_grad to build the target list.
        # posteriors_ref[t].mean/.std are used as fixed KL targets in 2nd pass.
        posteriors_ref: list[RSSMState] = []
        if overshoot_horizon > 0 and overshoot_weight > 0.0:
            prev_s = wm.rssm.initial(B, self._device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self._use_amp):
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

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(
                    prev_state, act_t, embed,
                )

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
                    img_t,       # preprocessed image — same space as decoder output
                    recon_img,
                    st_t,
                    recon_state,
                    free_nats=cfg.free_nats,
                    kl_dyn_scale=cfg.kl_dyn_scale,
                    kl_rep_scale=cfg.kl_rep_scale,
                    token_kl_weighting=token_kl_weighting,
                )

                # A1: warp smoothness regularization (set by last img_step call)
                if warp_smoothness_weight > 0 and hasattr(wm.rssm, "warp_smoothness_loss"):
                    smooth = wm.rssm.warp_smoothness_loss()
                    loss = loss + warp_smoothness_weight * smooth

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
                osh_kl_val = 0.0
                if posteriors_ref and t + 1 < T:
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

            prev_state = type(post)(
                *[x.detach() for x in post],
            )

        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
            self._optimizer.step()

        return {k: v / T for k, v in accum.items()} | {"total_loss": total_loss_value}

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
