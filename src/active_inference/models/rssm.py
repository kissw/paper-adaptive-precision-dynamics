from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class RSSMState(NamedTuple):
    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        stoch_dim: int = 64,
        deter_dim: int = 256,
        embed_dim: int = 256,
        action_dim: int = 2,
        hidden_dim: int = 256,
        min_std: float = 0.1,
        logvar_clip_low: float = -20.0,
        logvar_clip_high: float = 2.0,
    ):
        super().__init__()
        self._stoch_dim = stoch_dim
        self._deter_dim = deter_dim
        self._min_std = min_std
        self._logvar_clip_low = logvar_clip_low
        self._logvar_clip_high = logvar_clip_high

        # img_step: GRU input from (prev_stoch, prev_action)
        self._img_in = nn.Sequential(nn.Linear(stoch_dim + action_dim, hidden_dim), nn.ReLU())
        self._cell = nn.GRUCell(hidden_dim, deter_dim)

        # prior stats from deter
        self._img_out = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

        # posterior stats from (deter, embed)
        self._obs_out = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

    def initial(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        if device is None:
            device = next(self.parameters()).device
        z = torch.zeros(batch_size, self._stoch_dim, device=device)
        h = torch.zeros(batch_size, self._deter_dim, device=device)
        return RSSMState(deter=h, stoch=z, mean=z, std=torch.ones_like(z))

    def get_feat(self, state: RSSMState) -> Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def get_dist(self, state: RSSMState) -> Normal:
        return Normal(state.mean, state.std)

    def _stats(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_logvar = raw.chunk(2, dim=-1)
        logvar = raw_logvar.clamp(self._logvar_clip_low, self._logvar_clip_high)
        std = logvar.mul(0.5).exp() + self._min_std
        return mean, std

    def img_step(self, prev_state: RSSMState, prev_action: Tensor) -> RSSMState:
        x = self._img_in(torch.cat([prev_state.stoch, prev_action], dim=-1))
        h = self._cell(x, prev_state.deter)
        mean, std = self._stats(self._img_out(h))
        stoch = Normal(mean, std).rsample()
        return RSSMState(deter=h, stoch=stoch, mean=mean, std=std)

    def obs_step(
        self, prev_state: RSSMState, prev_action: Tensor, embed: Tensor
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev_state, prev_action)
        mean, std = self._stats(self._obs_out(torch.cat([prior.deter, embed], dim=-1)))
        stoch = Normal(mean, std).rsample()
        post = RSSMState(deter=prior.deter, stoch=stoch, mean=mean, std=std)
        return post, prior

    def imagine(self, initial_state: RSSMState, actions: Tensor) -> list[RSSMState]:
        # actions: [horizon, B, action_dim]
        states: list[RSSMState] = []
        state = initial_state
        for t in range(actions.shape[0]):
            state = self.img_step(state, actions[t])
            states.append(state)
        return states
