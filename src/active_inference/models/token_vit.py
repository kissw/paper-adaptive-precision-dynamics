"""Token-level ViT world model: encoder, transition, and decoders.

Spatial token dynamics are preserved end-to-end through the transition.
The transition operates on a (B, N, D) token grid and never collapses
tokens to a global vector internally.

State layout uses 2D flattened tensors for iCEM planner compatibility.
The planner's `x.expand(n_samples, -1)` pattern requires all state
fields to be 2D, so token dims are folded into the last axis:

  deter : (B, N*D_deter)  — flattened token deterministic context
  stoch : (B, N*Z)        — flattened token stochastic samples
  mean  : (B, Z)          — pooled mean  (EFE/GMM interface)
  std   : (B, Z)          — pooled std   (EFE/GMM interface)

All spatial operations reshape internally; pooling for the legacy
(B, feat_dim) interface is isolated to get_feat() and the state decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from active_inference.models.rssm import RSSMState


# ---------------------------------------------------------------------------
# TokenViTEncoder
# ---------------------------------------------------------------------------

class TokenViTEncoder(nn.Module):
    """Per-frame spatial ViT encoder: (B,3,H,W) + (B,S) -> (B,N,E).

    Produces one 256-dim embedding per spatial patch token. State is
    injected additively so navigation context modulates every token
    before the transformer contextualizes across patches.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 256,
        state_dim: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self._num_tokens = (image_size // patch_size) ** 2  # 64
        self._embed_dim = embed_dim

        # Patch embedding: each patch becomes one token.
        self._patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

        # Learned positional embedding, shared with transition.
        self._pos_embed = nn.Parameter(
            torch.zeros(1, self._num_tokens, embed_dim),
        )

        # State conditioning: project navigation state to embed_dim.
        self._state_mlp = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Spatial transformer: contextualizes token observations.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self._pos_embed, std=0.02)

    def forward(self, image: Tensor, state: Tensor) -> Tensor:
        # image: (B, 3, H, W)  state: (B, S)
        x = self._patch_embed(image)              # (B, E, G, G)
        x = x.flatten(2).transpose(1, 2)          # (B, N, E)
        x = x + self._pos_embed                   # (B, N, E)
        state_emb = self._state_mlp(state)        # (B, E)
        x = x + state_emb.unsqueeze(1)            # (B, N, E) broadcast
        return self._transformer(x)               # (B, N, E)


# ---------------------------------------------------------------------------
# TokenViTTransition
# ---------------------------------------------------------------------------

class TokenViTTransition(nn.Module):
    """Token-level stochastic transition — RSSM-interface compatible.

    Implements the same five methods as RSSM so planning, training, and
    evaluation code require no changes.

    Prior  (img_step):  prev tokens + action -> new deter + prior stoch
    Posterior (obs_step): prior deter + obs tokens -> posterior stoch

    The token grid is never globally pooled inside the transition.
    Pooling is isolated to get_feat() for backward-compatible EFE/GMM.
    """

    def __init__(
        self,
        num_tokens: int = 64,
        embed_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 64,
        action_dim: int = 2,
        num_prior_layers: int = 2,
        num_post_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        min_std: float = 0.1,
    ):
        super().__init__()
        self._N = num_tokens
        self._D = deter_dim
        self._Z = stoch_dim
        self._min_std = min_std

        # Positional embedding (shared across prior and posterior).
        self._pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, deter_dim),
        )

        # Project concat(prev_deter, prev_stoch) -> deter_dim.
        self._in_proj = nn.Linear(deter_dim + stoch_dim, deter_dim)

        # Action injection: broadcast action embedding to all tokens.
        self._action_mlp = nn.Sequential(
            nn.Linear(action_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, deter_dim),
        )

        # Prior transformer: deterministic token context update.
        prior_layer = nn.TransformerEncoderLayer(
            d_model=deter_dim,
            nhead=num_heads,
            dim_feedforward=int(deter_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self._prior_transformer = nn.TransformerEncoder(
            prior_layer, num_layers=num_prior_layers,
        )

        # Prior stochastic head: outputs token-level mean and raw std.
        self._prior_head = nn.Linear(deter_dim, stoch_dim * 2)

        # Posterior fusion: fuse prior_deter tokens with obs_tokens.
        self._obs_fuse = nn.Linear(deter_dim + embed_dim, deter_dim)

        # Posterior transformer.
        post_layer = nn.TransformerEncoderLayer(
            d_model=deter_dim,
            nhead=num_heads,
            dim_feedforward=int(deter_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self._post_transformer = nn.TransformerEncoder(
            post_layer, num_layers=num_post_layers,
        )

        # Posterior stochastic head.
        self._post_head = nn.Linear(deter_dim, stoch_dim * 2)

        # get_feat adapter: mean+max pooling -> linear projection.
        feat_dim = deter_dim + stoch_dim  # 320
        self._feat_adapter = nn.Linear(feat_dim * 2, feat_dim)

        nn.init.trunc_normal_(self._pos_embed, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stoch_from_raw(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split (B,N,Z*2) into token_mean, token_std, sample."""
        token_mean, raw_std = raw.chunk(2, dim=-1)          # each (B,N,Z)
        token_std = F.softplus(raw_std) + self._min_std     # (B,N,Z)
        stoch = Normal(token_mean, token_std).rsample()     # (B,N,Z)
        return token_mean, token_std, stoch

    def _pool_stats(self, token_mean: Tensor, token_std: Tensor) -> tuple[Tensor, Tensor]:
        """Pool (B,N,Z) token stats to (B,Z) for EFE/GMM interface."""
        return token_mean.mean(dim=1), token_std.mean(dim=1)

    def _make_state(
        self,
        deter: Tensor,       # (B, N, D)
        token_mean: Tensor,  # (B, N, Z)
        token_std: Tensor,   # (B, N, Z)
        stoch: Tensor,       # (B, N, Z)
    ) -> RSSMState:
        B = deter.shape[0]
        pooled_mean, pooled_std = self._pool_stats(token_mean, token_std)
        return RSSMState(
            deter=deter.reshape(B, self._N * self._D),
            stoch=stoch.reshape(B, self._N * self._Z),
            mean=pooled_mean,   # (B, Z) — for EFE/GMM
            std=pooled_std,     # (B, Z) — for EFE/GMM
        )

    # ------------------------------------------------------------------
    # RSSM-compatible interface
    # ------------------------------------------------------------------

    def initial(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        if device is None:
            device = next(self.parameters()).device
        return RSSMState(
            deter=torch.zeros(batch_size, self._N * self._D, device=device),
            stoch=torch.zeros(batch_size, self._N * self._Z, device=device),
            mean=torch.zeros(batch_size, self._Z, device=device),
            std=torch.ones(batch_size, self._Z, device=device),
        )

    def get_feat(self, state: RSSMState) -> Tensor:
        """Pool token features to (B, feat_dim=320) for legacy interfaces.

        Uses mean+max concatenation to preserve localized token evidence
        (e.g., a single obstacle token) that mean pooling would dilute.
        Only called by EFE scorer, iCEM, and state/image decoders that
        expect the legacy (B, 320) interface.
        """
        B = state.deter.shape[0]
        deter = state.deter.view(B, self._N, self._D)    # (B, N, D)
        stoch = state.stoch.view(B, self._N, self._Z)    # (B, N, Z)
        tok = torch.cat([deter, stoch], dim=-1)          # (B, N, 320)
        mean_p = tok.mean(dim=1)                          # (B, 320)
        max_p = tok.max(dim=1).values                     # (B, 320)
        return self._feat_adapter(
            torch.cat([mean_p, max_p], dim=-1)            # (B, 640)
        )                                                  # (B, 320)

    def get_dist(self, state: RSSMState) -> Normal:
        return Normal(state.mean, state.std)

    def img_step(self, prev_state: RSSMState, prev_action: Tensor) -> RSSMState:
        """Prior transition: propagate token dynamics without observations."""
        B = prev_state.deter.shape[0]
        deter = prev_state.deter.view(B, self._N, self._D)   # (B, N, D)
        stoch = prev_state.stoch.view(B, self._N, self._Z)   # (B, N, Z)

        # Project concat(prev_deter, prev_stoch) -> D.
        x = self._in_proj(torch.cat([deter, stoch], dim=-1)) # (B, N, D)
        x = x + self._pos_embed                               # (B, N, D)

        # Broadcast action embedding to every token.
        act_emb = self._action_mlp(prev_action)               # (B, D)
        x = x + act_emb.unsqueeze(1)                          # (B, N, D)

        # Prior transformer: new deterministic context.
        new_deter = self._prior_transformer(x)                # (B, N, D)

        # Stochastic prior head.
        raw = self._prior_head(new_deter)                     # (B, N, Z*2)
        token_mean, token_std, stoch_new = self._stoch_from_raw(raw)

        return self._make_state(new_deter, token_mean, token_std, stoch_new)

    def obs_step(
        self,
        prev_state: RSSMState,
        prev_action: Tensor,
        embed: Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        """Posterior update: fuse prior transition with observation tokens.

        embed: (B, N, embed_dim) from TokenViTEncoder.
        Returns (posterior, prior).
        """
        prior = self.img_step(prev_state, prev_action)

        B = prior.deter.shape[0]
        prior_deter = prior.deter.view(B, self._N, self._D)  # (B, N, D)

        # Fuse prior deter tokens with observation tokens.
        fused = self._obs_fuse(
            torch.cat([prior_deter, embed], dim=-1)          # (B, N, D+E)
        )                                                     # (B, N, D)
        fused = fused + self._pos_embed                      # (B, N, D)

        # Posterior transformer.
        post_x = self._post_transformer(fused)               # (B, N, D)

        # Posterior stochastic head.
        raw = self._post_head(post_x)                        # (B, N, Z*2)
        token_mean, token_std, stoch_post = self._stoch_from_raw(raw)

        # Posterior shares prior's deter (RSSM convention).
        post = self._make_state(prior_deter, token_mean, token_std, stoch_post)
        return post, prior

    def imagine(self, initial_state: RSSMState, actions: Tensor) -> list[RSSMState]:
        """Open-loop rollout under action sequence.

        actions: (H, B, action_dim)
        Returns list of H RSSMState objects.
        """
        states: list[RSSMState] = []
        state = initial_state
        for t in range(actions.shape[0]):
            state = self.img_step(state, actions[t])
            states.append(state)
        return states


# ---------------------------------------------------------------------------
# TokenImageDecoder
# ---------------------------------------------------------------------------

class TokenImageDecoder(nn.Module):
    """Decode token grid back to image: (B,N*D,B,N*Z) -> (B,3,64,64).

    Reshapes flat token state to spatial grid, then uses strided
    transposed convolutions to upsample from 8x8 to 64x64.
    """

    def __init__(
        self,
        num_tokens: int = 64,
        deter_dim: int = 256,
        stoch_dim: int = 64,
        dec_dim: int = 256,
        image_channels: int = 3,
    ):
        super().__init__()
        self._num_tokens = num_tokens
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        grid_side = int(num_tokens ** 0.5)   # 8
        assert grid_side * grid_side == num_tokens

        feat_dim = deter_dim + stoch_dim  # 320

        # Project token features to dec_dim.
        self._tok_proj = nn.Sequential(
            nn.Linear(feat_dim, dec_dim),
            nn.ReLU(),
        )

        # Upsample 8x8 -> 64x64 via three 2x ConvTranspose stages.
        self._deconv = nn.Sequential(
            nn.ConvTranspose2d(dec_dim, 128, 4, stride=2, padding=1),  # 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),       # 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),        # 64
            nn.ReLU(),
            nn.Conv2d(32, image_channels, 3, padding=1),               # 64
        )

    def forward(self, deter_flat: Tensor, stoch_flat: Tensor) -> Tensor:
        # deter_flat: (B, N*D_deter)  stoch_flat: (B, N*Z)
        B = deter_flat.shape[0]
        deter = deter_flat.view(B, self._num_tokens, self._deter_dim)
        stoch = stoch_flat.view(B, self._num_tokens, self._stoch_dim)
        tok = torch.cat([deter, stoch], dim=-1)              # (B, N, 320)
        tok = self._tok_proj(tok)                             # (B, N, dec_dim)
        grid_side = int(self._num_tokens ** 0.5)             # 8
        # Reshape to spatial grid and upsample.
        grid = tok.permute(0, 2, 1).view(
            B, -1, grid_side, grid_side,
        )                                                     # (B, dec_dim, 8, 8)
        return self._deconv(grid)                             # (B, 3, 64, 64)


# ---------------------------------------------------------------------------
# TokenStateDecoder
# ---------------------------------------------------------------------------

class TokenStateDecoder(nn.Module):
    """Decode pooled token feature to navigation state vector.

    Input: (B, feat_dim=320) from get_feat().
    Output: (B, state_dim=4) = [speed, steer, heading_err, crosstrack_err].
    """

    def __init__(self, feat_dim: int = 320, state_dim: int = 4):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self._mlp(feat)
