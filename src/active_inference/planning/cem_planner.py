from __future__ import annotations

import torch
from torch import Tensor


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
    ):
        self._action_dim = action_dim
        self._horizon = horizon
        self._n_samples = n_samples
        self._n_elites = n_elites
        self._n_iters = n_iters
        self._beta = colored_noise_beta
        self._warm_start = warm_start
        self._prev_mean: Tensor | None = None

    def reset(self):
        self._prev_mean = None

    @torch.no_grad()
    def plan(self, initial_state, rssm, efe_scorer, pref_model, ensemble) -> Tensor:
        device = initial_state.deter.device

        if self._warm_start and self._prev_mean is not None:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
            mean[:-1] = self._prev_mean[1:]
        else:
            mean = torch.zeros(self._horizon, self._action_dim, device=device)
        std = torch.ones(self._horizon, self._action_dim, device=device)

        for _ in range(self._n_iters):
            noise = colored_noise(
                (self._n_samples, self._horizon, self._action_dim), self._beta
            ).to(device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)

            # Batch-parallel rollout: expand initial state to N samples
            expanded = type(initial_state)(*[x.expand(self._n_samples, -1) for x in initial_state])
            # actions [N, H, A] -> [H, N, A]
            trajectory = rssm.imagine(expanded, actions.permute(1, 0, 2))

            feats = [rssm.get_feat(s) for s in trajectory]
            means_list = [s.mean for s in trajectory]
            stds_list = [s.std for s in trajectory]

            scores = efe_scorer.score(feats, means_list, stds_list, pref_model, ensemble)

            elite_idxs = scores.argsort()[: self._n_elites]
            elite_actions = actions[elite_idxs]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-5

        self._prev_mean = mean.detach()
        return mean[0].clamp(-1.0, 1.0)
