from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class PlanResult(NamedTuple):
    action: Tensor  # [action_dim]
    efe_score: float  # best trajectory EFE
    epistemic_score: float  # best trajectory epistemic


def colored_noise(shape: tuple[int, int, int], beta: float = 1.0) -> Tensor:
    # FFT-based 1/f^beta noise for temporally smooth actions
    n_samples, horizon, action_dim = shape
    freqs = torch.fft.rfftfreq(horizon)
    freqs[0] = 1.0
    power = 1.0 / freqs.pow(beta / 2.0)
    power[0] = 0.0
    white = torch.randn(n_samples, horizon, action_dim)
    spectrum = torch.fft.rfft(white, dim=1)
    result = torch.fft.irfft(spectrum * power.unsqueeze(0).unsqueeze(-1), n=horizon, dim=1)
    return result / (result.std(dim=1, keepdim=True) + 1e-8)


class iCEMPlanner:
    def __init__(
        self,
        action_dim: int = 2,
        horizon: int = 12,
        n_samples: int = 200,
        n_elites: int = 20,
        n_iters: int = 3,
        colored_noise_beta: float = 1.0,
        warm_start: bool = True,
        accel_prior: float = 0.3,
        noise_scale: list[float] | None = None,
        cold_start_extra_iters: int = 5,
        min_std: float = 0.1,
        keep_fraction: float = 0.1,
        warm_start_reset_threshold: float = 1.0,
    ):
        self._action_dim = action_dim
        self._horizon = horizon
        self._n_samples = n_samples
        self._n_elites = n_elites
        self._n_iters = n_iters
        self._beta = colored_noise_beta
        self._warm_start = warm_start
        self._accel_prior = accel_prior
        self._noise_scale = torch.tensor(noise_scale) if noise_scale is not None else None
        self._cold_start_extra_iters = cold_start_extra_iters
        self._min_std = min_std
        self._keep_fraction = keep_fraction
        self._warm_start_reset_threshold = warm_start_reset_threshold
        self._prev_mean: Tensor | None = None
        self._adaptive_min_std = min_std

    def reset(self):
        self._prev_mean = None
        self._adaptive_min_std = self._min_std

    def set_state_context(
        self,
        heading_error: float,
        crosstrack_error: float,
        state_error: float,
    ):
        """Adapt exploration and action prior from observed state error.

        Two AIF mechanisms:
        1. Precision-weighting: high prediction error reduces precision
           on the action prior, widening exploration (wider min_std).
        2. Corrective action prior: heading/crosstrack error sets an
           informed prior on steering (proprioceptive reflex). The CEM
           optimizes around this prior rather than the warm-start mean.
        """
        self._heading_error = heading_error
        self._crosstrack_error = crosstrack_error
        self._state_error = state_error

        # Exploration widening (threshold configurable for Task B)
        if state_error > self._warm_start_reset_threshold:
            self._adaptive_min_std = self._min_std * 3.0
            self._prev_mean = None
        elif state_error > 0.3:
            boost = 1.0 + 2.0 * (state_error - 0.3) / 0.7
            self._adaptive_min_std = self._min_std * boost
        else:
            self._adaptive_min_std = self._min_std

    @torch.no_grad()
    def plan(self, initial_state, rssm, efe_scorer, pref_model, ensemble, state_decoder=None, obstacle_info=None) -> PlanResult:
        device = initial_state.deter.device

        # One-time warm-start reset when obstacle enters detection zone
        # to break inertia from straight-driving trajectory
        if obstacle_info is not None and obstacle_info.get("reset_warmstart", False):
            self._prev_mean = None
            self._adaptive_min_std = self._min_std * 2.0

        if self._warm_start and self._prev_mean is not None:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
            mean[:-1] = self._prev_mean[1:]
        else:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
            # Acceleration prior: break cold-start symmetry so CEM can
            # differentiate moving vs stopped trajectories.
            if self._accel_prior != 0.0 and self._action_dim >= 2:
                mean[:, 1] = self._accel_prior

        # AIF action prior modulation: obstacle observation shifts the
        # steer prior from neutral to directional evasion.
        # This is analogous to a reflexive motor prior — seeing an obstacle
        # biases the action distribution toward the escape direction.
        if obstacle_info is not None:
            evasion_steer = obstacle_info.get("evasion_steer", 0.0)
            obs_prox = obstacle_info.get("proximity", 0.0)
            if evasion_steer != 0.0 and obs_prox > 0.05:
                # Scale bias by proximity: stronger as obstacle gets closer
                bias_strength = evasion_steer * min(obs_prox * 2.0, 1.0)
                # Apply sustained bias across all horizon steps
                mean[:, 0] = mean[:, 0] + bias_strength
                # Also reduce throttle prior when evading
                if self._action_dim >= 2:
                    mean[:, 1] = mean[:, 1] - 0.15 * min(obs_prox * 2.0, 1.0)
                mean = mean.clamp(-1.0, 1.0)

        std = torch.ones(self._horizon, self._action_dim, device=device)

        # Cold-start: use extra iterations when no warm-start prior exists.
        # More CEM iterations help find the good action basin from scratch.
        is_cold = not (self._warm_start and self._prev_mean is not None)
        iters = self._n_iters + (self._cold_start_extra_iters if is_cold else 0)

        # Cold-start: use more samples to better cover the action space
        n_total = self._n_samples * 2 if is_cold else self._n_samples
        n_keep = max(1, int(n_total * self._keep_fraction))

        # Lane-change motor primitives: when obstacle detected with evasion
        # direction, inject trajectory templates with sustained directional
        # steer into the CEM population. This ensures the planner always
        # has lane-change candidates for the penalty to select as elites.
        # In AIF terms: motor primitives are pre-learned action sequences
        # in the action prior repertoire.
        n_templates = 0
        template_actions = None
        if obstacle_info is not None:
            ev_s = obstacle_info.get("evasion_steer", 0.0)
            ev_p = obstacle_info.get("proximity", 0.0)
            if ev_s != 0.0 and ev_p > 0.05:
                n_templates = max(1, int(n_total * 0.15))
                template_actions = torch.zeros(n_templates, self._horizon, self._action_dim, device=device)
                # S-curve lane change: steer → coast → counter-steer
                H = self._horizon
                phase1 = H // 3         # initiation
                phase2 = 2 * H // 3     # coast
                for i in range(n_templates):
                    # Vary steer magnitude across templates
                    mag = 0.3 + 0.4 * (i / max(n_templates - 1, 1))
                    sign = 1.0 if ev_s > 0 else -1.0
                    template_actions[i, :phase1, 0] = sign * mag
                    template_actions[i, phase1:phase2, 0] = sign * mag * 0.3
                    template_actions[i, phase2:, 0] = -sign * mag * 0.2
                    # Moderate throttle (need speed for lane change)
                    template_actions[i, :, 1] = 0.2
                template_actions = template_actions.clamp(-1.0, 1.0)

        n_from_dist = n_total - n_keep - n_templates

        for _ in range(iters):
            # Main population from elite distribution
            noise = colored_noise(
                (n_from_dist, self._horizon, self._action_dim), self._beta
            ).to(device)
            if self._noise_scale is not None:
                noise = noise * self._noise_scale.to(device)
            dist_actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)

            # Population injection: random samples to prevent variance collapse
            rand_noise = colored_noise(
                (n_keep, self._horizon, self._action_dim), self._beta
            ).to(device)
            if self._noise_scale is not None:
                rand_noise = rand_noise * self._noise_scale.to(device)
            rand_actions = rand_noise.clamp(-1.0, 1.0)

            parts = [dist_actions, rand_actions]
            if template_actions is not None:
                parts.append(template_actions)
            actions = torch.cat(parts, dim=0)

            # Batch-parallel rollout: expand initial state to match action count
            n_actions = actions.shape[0]
            expanded = type(initial_state)(*[x.expand(n_actions, -1) for x in initial_state])
            # actions [N, H, A] -> [H, N, A]
            trajectory = rssm.imagine(expanded, actions.permute(1, 0, 2))

            feats = [rssm.get_feat(s) for s in trajectory]
            means_list = [s.mean for s in trajectory]
            stds_list = [s.std for s in trajectory]

            scores = efe_scorer.score(feats, means_list, stds_list, pref_model, ensemble, state_decoder, obstacle_info)

            # Action-prior obstacle avoidance: directional evasion penalty.
            # In AIF terms: obstacle observation modulates the action prior
            # from "prefer small steer" to "prefer steering in evasion direction".
            # The penalty is DIRECTIONAL — it specifically rewards trajectories
            # that steer toward the correct evasion side and penalizes wrong-way
            # or straight trajectories. This breaks the left-right symmetry that
            # causes CEM to oscillate.
            if obstacle_info is not None:
                obs_prox = obstacle_info.get("proximity", 0.0)
                beta_o = obstacle_info.get("beta_obstacle", 3.0)
                evasion_steer = obstacle_info.get("evasion_steer", 0.0)
                if obs_prox > 0:
                    H = actions.shape[1]  # horizon

                    if evasion_steer != 0.0:
                        # Directional penalty: penalize NOT steering in evasion direction
                        evasion_sign = 1.0 if evasion_steer > 0 else -1.0
                        cumul_steer = actions[:, :, 0].cumsum(dim=1)
                        # Project final cumulative steer onto evasion direction
                        final_evasion = cumul_steer[:, -1] * evasion_sign  # positive = correct
                        # Score: 0→1 as cumulative steer reaches 3.0 in correct direction
                        evasion_score = (final_evasion / 3.0).clamp(min=0.0, max=1.0)
                        directional_penalty = (1.0 - evasion_score).pow(2)
                    else:
                        # Undirected fallback: penalize low absolute steer
                        cumul_steer = actions[:, :, 0].cumsum(dim=1)
                        max_lateral = cumul_steer.abs().max(dim=1).values
                        mean_abs_steer = actions[:, :, 0].abs().mean(dim=1)
                        lateral_score = (max_lateral / 0.5).clamp(max=1.0)
                        sustained_score = (mean_abs_steer / 0.15).clamp(max=1.0)
                        avoidance_score = (lateral_score * sustained_score).clamp(max=1.0)
                        directional_penalty = (1.0 - avoidance_score).pow(2)

                    # Throttle penalty: penalize high acceleration near obstacles
                    mean_accel = actions[:, :, 1].mean(dim=1)
                    speed_penalty = (mean_accel.clamp(min=0.0) / 0.3).clamp(max=1.0)

                    combined = directional_penalty + 0.2 * speed_penalty
                    scores = scores + obs_prox * beta_o * H * combined

            elite_idxs = scores.argsort()[: self._n_elites]
            elite_actions = actions[elite_idxs]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=self._adaptive_min_std)

            # Re-inject evasion bias after elite update to prevent wash-out.
            # The directional penalty selects correct-direction elites, and
            # this bias keeps the mean shifted toward the evasion direction
            # across CEM iterations (persistent action prior).
            if obstacle_info is not None:
                ev_s = obstacle_info.get("evasion_steer", 0.0)
                ev_p = obstacle_info.get("proximity", 0.0)
                if ev_s != 0.0 and ev_p > 0.05:
                    bias = ev_s * min(ev_p * 2.0, 1.0) * 0.5
                    mean[:, 0] = (mean[:, 0] + bias).clamp(-1.0, 1.0)

        self._prev_mean = mean.detach()

        # Score the elite mean trajectory (1 sample) to extract EFE/epistemic
        elite_mean_actions = mean.unsqueeze(0).clamp(-1.0, 1.0)  # [1, H, A]
        expanded_one = type(initial_state)(*[x[:1] for x in initial_state])
        traj_one = rssm.imagine(expanded_one, elite_mean_actions.permute(1, 0, 2))
        feats_one = [rssm.get_feat(s) for s in traj_one]
        means_one = [s.mean for s in traj_one]
        stds_one = [s.std for s in traj_one]

        efe_total = efe_scorer.score(feats_one, means_one, stds_one, pref_model, ensemble, state_decoder, obstacle_info)
        epistemic_total = sum(
            ensemble.epistemic_uncertainty(f).item() for f in feats_one
        )

        return PlanResult(
            action=mean[0].clamp(-1.0, 1.0),
            efe_score=float(efe_total.item()),
            epistemic_score=float(epistemic_total),
        )
