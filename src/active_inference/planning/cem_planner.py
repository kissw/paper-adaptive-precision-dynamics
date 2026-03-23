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

        # Exploration widening
        if state_error > 1.0:
            self._adaptive_min_std = self._min_std * 3.0
            self._prev_mean = None
        elif state_error > 0.3:
            boost = 1.0 + 2.0 * (state_error - 0.3) / 0.7
            self._adaptive_min_std = self._min_std * boost
        else:
            self._adaptive_min_std = self._min_std

    @torch.no_grad()
    def plan(self, initial_state, rssm, efe_scorer, pref_model, ensemble, state_decoder=None) -> PlanResult:
        device = initial_state.deter.device

        if self._warm_start and self._prev_mean is not None:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
            mean[:-1] = self._prev_mean[1:]
        else:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
            # Acceleration prior: break cold-start symmetry so CEM can
            # differentiate moving vs stopped trajectories.
            if self._accel_prior != 0.0 and self._action_dim >= 2:
                mean[:, 1] = self._accel_prior
        std = torch.ones(self._horizon, self._action_dim, device=device)

        # Cold-start: use extra iterations when no warm-start prior exists.
        # More CEM iterations help find the good action basin from scratch.
        is_cold = not (self._warm_start and self._prev_mean is not None)
        iters = self._n_iters + (self._cold_start_extra_iters if is_cold else 0)

        # Cold-start: use more samples to better cover the action space
        n_total = self._n_samples * 2 if is_cold else self._n_samples
        n_keep = max(1, int(n_total * self._keep_fraction))
        n_from_dist = n_total - n_keep

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

            actions = torch.cat([dist_actions, rand_actions], dim=0)

            # Batch-parallel rollout: expand initial state to N samples
            expanded = type(initial_state)(*[x.expand(n_total, -1) for x in initial_state])
            # actions [N, H, A] -> [H, N, A]
            trajectory = rssm.imagine(expanded, actions.permute(1, 0, 2))

            feats = [rssm.get_feat(s) for s in trajectory]
            means_list = [s.mean for s in trajectory]
            stds_list = [s.std for s in trajectory]

            scores = efe_scorer.score(feats, means_list, stds_list, pref_model, ensemble, state_decoder)

            elite_idxs = scores.argsort()[: self._n_elites]
            elite_actions = actions[elite_idxs]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=self._adaptive_min_std)

        self._prev_mean = mean.detach()

        # Score the elite mean trajectory (1 sample) to extract EFE/epistemic
        elite_mean_actions = mean.unsqueeze(0).clamp(-1.0, 1.0)  # [1, H, A]
        expanded_one = type(initial_state)(*[x[:1] for x in initial_state])
        traj_one = rssm.imagine(expanded_one, elite_mean_actions.permute(1, 0, 2))
        feats_one = [rssm.get_feat(s) for s in traj_one]
        means_one = [s.mean for s in traj_one]
        stds_one = [s.std for s in traj_one]

        efe_total = efe_scorer.score(feats_one, means_one, stds_one, pref_model, ensemble, state_decoder)
        epistemic_total = sum(
            ensemble.epistemic_uncertainty(f).item() for f in feats_one
        )

        return PlanResult(
            action=mean[0].clamp(-1.0, 1.0),
            efe_score=float(efe_total.item()),
            epistemic_score=float(epistemic_total),
        )
