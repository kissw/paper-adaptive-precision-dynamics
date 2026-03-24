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
from active_inference.training.losses import compute_vfe
from active_inference.training.preference import PreferenceModel


class WorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ConvEncoder(
            image_channels=cfg.encoder.image_channels,
            state_dim=cfg.encoder.state_dim,
            embed_dim=cfg.rssm.embed_dim,
            crop_road=getattr(cfg.encoder, "crop_road", False),
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
        self.ensemble = EnsembleTransitionHeads(
            feat_dim=feat_dim,
            stoch_dim=cfg.rssm.stoch_dim,
            hidden_dim=cfg.ensemble.hidden_dim,
            num_heads=cfg.ensemble.num_heads,
        )


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
    def step_with_info(self, obs_img: Tensor, obs_state: Tensor) -> PlanResult:
        if self._prev_state is None:
            self.reset()

        # Full 4D state [speed, steer, heading_error, crosstrack_error] to encoder
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

        embed = self.world_model.encoder(img, st)
        post, _ = self.world_model.rssm.obs_step(self._prev_state, self._prev_action, embed)

        # AIF precision-weighting: observed state error modulates
        # both action prior precision and EFE channel weights.
        # High error → wider exploration + trust state decoder over preference.
        heading_err = float(st[0, 2]) if st.dim() == 2 else float(st[2])
        crosstrack_err = float(st[0, 3]) if st.dim() == 2 else float(st[3])
        state_error = heading_err ** 2 + crosstrack_err ** 2
        self.planner.set_state_context(heading_err, crosstrack_err, state_error)

        # Adaptive EFE precision: when state error is high, the preference
        # model operates on OOD latents (it was trained on centered driving).
        # Increase state penalty weight (direct observation, reliable) and
        # decrease preference weight (model-based, unreliable at OOD states).
        if state_error > 0.3:
            boost = min(4.0, 1.0 + (state_error - 0.3) * 4.0)
            self.efe_scorer._beta_s = self._cfg.efe.beta_state * boost
            self.efe_scorer._beta_i = self._cfg.efe.beta_instrumental / boost
        else:
            self.efe_scorer._beta_s = self._cfg.efe.beta_state
            self.efe_scorer._beta_i = self._cfg.efe.beta_instrumental

        plan_result = self.planner.plan(
            post,
            self.world_model.rssm,
            self.efe_scorer,
            self.preference,
            self.world_model.ensemble,
            self.world_model.state_decoder,
        )

        self._prev_state = post
        self._prev_action = plan_result.action.unsqueeze(0)
        return plan_result

    @torch.no_grad()
    def step(self, obs_img: Tensor, obs_state: Tensor) -> Tensor:
        return self.step_with_info(obs_img, obs_state).action

    def update(self, images: Tensor, states: Tensor, actions: Tensor) -> dict[str, float]:
        # Per-timestep backward with NaN guard, AMP, and CUDA error recovery
        B, T = images.shape[0], images.shape[1]
        wm = self.world_model
        cfg = self._cfg.training

        self._optimizer.zero_grad()
        total_loss_value = 0.0
        accum = {"img_loss": 0.0, "state_loss": 0.0, "kl_dyn": 0.0, "kl_rep": 0.0}

        prev_state = wm.rssm.initial(B, self._device)
        prev_action = torch.zeros(B, self._cfg.cem.action_dim, device=self._device)

        for t in range(T):
            img_t = images[:, t].to(self._device)
            st_t = states[:, t].to(self._device)
            act_t = actions[:, t].to(self._device) if t > 0 else prev_action

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = wm.encoder(img_t, st_t)
                post, prior = wm.rssm.obs_step(prev_state, prev_action, embed)

                feat = wm.rssm.get_feat(post)
                recon_img = wm.obs_decoder(feat)
                recon_state = wm.state_decoder(feat)

                loss, info = compute_vfe(
                    post.mean,
                    post.std,
                    prior.mean,
                    prior.std,
                    img_t,
                    recon_img,
                    st_t,
                    recon_state,
                    free_nats=cfg.free_nats,
                    kl_dyn_scale=cfg.kl_dyn_scale,
                    kl_rep_scale=cfg.kl_rep_scale,
                )

            # NaN guard: skip this timestep if loss is bad
            if torch.isnan(loss) or torch.isinf(loss):
                prev_state = RSSMState(*[x.detach() for x in post])
                prev_action = act_t.detach() if t > 0 else prev_action
                continue

            scaled_loss = loss / T
            if self._scaler is not None:
                self._scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss_value += loss.item() / T

            for k in accum:
                accum[k] += info[k].item()

            prev_state = RSSMState(*[x.detach() for x in post])
            prev_action = act_t.detach() if t > 0 else prev_action

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
        self.world_model.load_state_dict(ckpt["world_model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        pref = ckpt["preference"]
        self.preference.means.data.copy_(pref["means"])
        self.preference.log_stds.data.copy_(pref["log_stds"])
        self.preference.logits.data.copy_(pref["logits"])

    @torch.no_grad()
    def encode_preference_data(self, pref_loader, max_samples: int = 3000) -> Tensor:
        latents = []
        wm = self.world_model
        count = 0

        for images, states, actions in pref_loader:
            B, T = images.shape[:2]
            state = wm.rssm.initial(B, self._device)
            prev_act = torch.zeros(B, self._cfg.cem.action_dim, device=self._device)

            for t in range(T):
                embed = wm.encoder(
                    images[:, t].to(self._device),
                    states[:, t].to(self._device),
                )
                post, _ = wm.rssm.obs_step(state, prev_act, embed)
                latents.append(post.mean.cpu())
                state = RSSMState(*[x.detach() for x in post])
                prev_act = actions[:, t].to(self._device)

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
