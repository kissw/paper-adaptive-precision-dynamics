from __future__ import annotations

from pathlib import Path

import numpy as np
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
        )
        self.efe_scorer = EFEScorer(
            beta_instrumental=cfg.efe.beta_instrumental,
            beta_epistemic=cfg.efe.beta_epistemic,
            mc_samples=cfg.efe.mc_samples,
        )

        self._optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.training.lr)
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

        # Extract steering signals before slicing (not fed to model).
        heading_error = None
        crosstrack_error = 0.0
        if obs_state.shape[-1] >= 4:
            heading_error = float(obs_state[..., 2])
            crosstrack_error = float(obs_state[..., 3])
        elif obs_state.shape[-1] >= 3:
            heading_error = float(obs_state[..., 2])

        # Slice state to match encoder's expected state_dim (handles 3D env → 2D model)
        expected_dim = self._cfg.encoder.state_dim
        if obs_state.shape[-1] > expected_dim:
            obs_state = obs_state[..., :expected_dim]

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

        plan_result = self.planner.plan(
            post,
            self.world_model.rssm,
            self.efe_scorer,
            self.preference,
            self.world_model.ensemble,
        )

        # Post-CEM steering override: Stanley controller using road heading
        # error + crosstrack correction. CEM handles longitudinal (speed)
        # control via EFE; lateral uses classical Stanley feedback.
        if heading_error is not None:
            speed = float(obs_state[..., 0]) if obs_state.shape[-1] >= 1 else 1.0
            # Stanley: steer = heading_error + arctan(k * crosstrack / (speed + eps))
            # Negative signs: heading_error>0 → road left → steer left (negative)
            #                 crosstrack>0 → vehicle right → steer left (negative)
            k_heading = 1.5
            k_crosstrack = 2.0
            # Positive steer = LEFT turn in CARLA (counterclockwise yaw increase).
            # heading_error > 0 → road is left → steer left (positive).
            # crosstrack > 0 → vehicle right of center → steer left (positive).
            stanley_steer = k_heading * heading_error + np.arctan2(
                k_crosstrack * crosstrack_error, max(speed, 0.5)
            )
            corrected_steer = max(-1.0, min(1.0, stanley_steer))
            corrected_action = plan_result.action.clone()
            corrected_action[0] = corrected_steer
            plan_result = PlanResult(
                corrected_action, plan_result.efe_score, plan_result.epistemic_score
            )

        self._prev_state = post
        self._prev_action = plan_result.action.unsqueeze(0)
        return plan_result

    @torch.no_grad()
    def step(self, obs_img: Tensor, obs_state: Tensor) -> Tensor:
        return self.step_with_info(obs_img, obs_state).action

    def update(self, images: Tensor, states: Tensor, actions: Tensor) -> dict[str, float]:
        # Slice state to match encoder's expected state_dim
        expected_dim = self._cfg.encoder.state_dim
        if states.shape[-1] > expected_dim:
            states = states[..., :expected_dim]

        # Per-timestep backward with NaN guard and CUDA error recovery
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

            (loss / T).backward()
            total_loss_value += loss.item() / T

            for k in accum:
                accum[k] += info[k].item()

            prev_state = RSSMState(*[x.detach() for x in post])
            prev_action = act_t.detach() if t > 0 else prev_action

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
