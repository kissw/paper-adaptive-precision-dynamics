"""Token-level ViT world model: encoder, transition, and decoders.

Spatial token dynamics are preserved end-to-end through the transition.
The transition operates on a (B, N, D) token grid and never collapses
tokens to a global vector internally.

Spatial token order (verified):
  Encoder: Conv2d(patch=8,stride=8) → (B,E,8,8) → flatten(2) → (B,E,64)
           → transpose(1,2) → (B,64,E).
  Token i maps to image patch at (row = i // 8, col = i % 8) — raster order.
  Decoder: permute(0,2,1) → (B,E,64) → reshape(B,E,8,8) uses the same mapping.
  ActionWarp (_apply_warp) follows the same permute+reshape convention so that
  spatial flow displacements correspond to the correct image regions.

State layout (TokenRSSMState — 6 fields):

  deter      : (B, N, D_deter)  — 3D token deterministic context
  stoch      : (B, N, Z)        — 3D token stochastic samples
  mean       : (B, Z)           — pooled mean  (EFE/GMM interface)
  std        : (B, Z)           — pooled std   (EFE/GMM interface)
  token_mean : (B, N, Z)        — per-token mean for VFE KL
  token_std  : (B, N, Z)        — per-token std  for VFE KL

EFE/GMM sees pooled (B, Z) mean/std so the preference model receives the
expected shape.  VFE uses token_mean/token_std so KL is computed token-wise
(KL.sum(-1).mean() handles both (B,Z) and (B,N,Z) shapes).

iCEM expand uses x.expand(n_samples, *x.shape[1:]) which is generic for
any number of trailing dims, so 3D deter/stoch fields work correctly.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# TokenRSSMState
# ---------------------------------------------------------------------------

class TokenRSSMState(NamedTuple):
    deter: Tensor       # (B, N, D) — 3D token deterministic context
    stoch: Tensor       # (B, N, Z) — 3D token stochastic sample
    mean: Tensor        # (B, Z)    — pooled, for EFE/GMM compatibility
    std: Tensor         # (B, Z)    — pooled, for EFE/GMM compatibility
    token_mean: Tensor  # (B, N, Z) — per-token, for VFE KL
    token_std: Tensor   # (B, N, Z) — per-token, for VFE KL


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
        use_action_warp: bool = False,
    ):
        super().__init__()
        self._N = num_tokens
        self._D = deter_dim
        self._Z = stoch_dim
        self._min_std = min_std
        self._use_action_warp = use_action_warp
        self._action_dim = action_dim

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

        # A1: Action-warp modules (only created when use_action_warp=True)
        if use_action_warp:
            assert int(num_tokens ** 0.5) ** 2 == num_tokens, \
                "use_action_warp requires num_tokens to be a perfect square"
            self._grid_size = int(num_tokens ** 0.5)   # 8
            self._flow_scale = 0.1                     # initial flow magnitude cap
            # Global flow: single displacement applied to all tokens
            self._warp_global = nn.Linear(action_dim, 2)
            nn.init.zeros_(self._warp_global.weight)
            nn.init.zeros_(self._warp_global.bias)
            # Per-token residual flow
            self._warp_residual = nn.Sequential(
                nn.Linear(deter_dim + action_dim, deter_dim),
                nn.SiLU(),
                nn.Linear(deter_dim, 2),
            )
            nn.init.zeros_(self._warp_residual[-1].weight)
            nn.init.zeros_(self._warp_residual[-1].bias)
            self._warp_smooth_loss: Tensor | None = None

        nn.init.trunc_normal_(self._pos_embed, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # A1: Action warp helpers
    # ------------------------------------------------------------------

    def _make_base_grid(self, G: int, device: torch.device) -> Tensor:
        """Return identity sampling grid (1, G, G, 2) for F.grid_sample.

        Last dim order: (x = col direction, y = row direction),
        values in [-1, 1] with align_corners=True convention.
        """
        lin = torch.linspace(-1, 1, G, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")  # (G, G) each
        base = torch.stack([gx, gy], dim=-1)              # (G, G, 2): (col, row)
        return base.unsqueeze(0)                           # (1, G, G, 2)

    def _apply_warp(self, x: Tensor, action: Tensor) -> Tensor:
        """Warp token grid by action-conditioned flow field.

        Follows the same (B,N,D) ↔ (B,D,G,G) convention as the decoder:
          permute(0,2,1).reshape(B,D,G,G)  →  grid_sample  →  reshape+permute back.

        Flow channels: (dx_col, dy_row) matching grid_sample's (x, y) order.
        """
        B, N, D = x.shape
        G = self._grid_size

        # (B, N, D) → (B, D, G, G): decoder-compatible reshape
        grid_feat = x.permute(0, 2, 1).reshape(B, D, G, G)

        # Global flow: same (dx, dy) for all tokens
        gflow = self._warp_global(action)                    # (B, 2)

        # Per-token residual: (B, N, D+action_dim) → (B, N, 2)
        act_exp = action.unsqueeze(1).expand(B, N, -1)       # (B, N, action_dim)
        rflow = self._warp_residual(
            torch.cat([x, act_exp], dim=-1)
        )                                                     # (B, N, 2)

        flow = gflow.unsqueeze(1) + rflow                    # (B, N, 2)
        flow_2d = flow.reshape(B, G, G, 2)                   # (B, G, G, 2)

        # Sampling grid: identity + scaled flow
        base = self._make_base_grid(G, x.device)             # (1, G, G, 2)
        samp = base + flow_2d * self._flow_scale             # (B, G, G, 2)

        warped = F.grid_sample(
            grid_feat, samp,
            align_corners=True, padding_mode="border",
        )                                                     # (B, D, G, G)

        # Store smoothness loss (with grad) before returning
        diff_h = (flow_2d[:, 1:] - flow_2d[:, :-1]).pow(2).mean()
        diff_w = (flow_2d[:, :, 1:] - flow_2d[:, :, :-1]).pow(2).mean()
        self._warp_smooth_loss = diff_h + diff_w

        # (B, D, G, G) → (B, N, D): reverse the reshape
        return warped.reshape(B, D, N).permute(0, 2, 1)

    def warp_smoothness_loss(self) -> Tensor:
        """Return TV-L2 flow smoothness penalty from the last img_step call."""
        if not self._use_action_warp or self._warp_smooth_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._warp_smooth_loss

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
    ) -> TokenRSSMState:
        pooled_mean, pooled_std = self._pool_stats(token_mean, token_std)
        return TokenRSSMState(
            deter=deter,
            stoch=stoch,
            mean=pooled_mean,         # (B, Z) — for EFE/GMM
            std=pooled_std,           # (B, Z) — for EFE/GMM
            token_mean=token_mean,    # (B, N, Z) — for VFE KL
            token_std=token_std,      # (B, N, Z) — for VFE KL
        )

    # ------------------------------------------------------------------
    # RSSM-compatible interface
    # ------------------------------------------------------------------

    def initial(self, batch_size: int, device: torch.device | None = None) -> TokenRSSMState:
        if device is None:
            device = next(self.parameters()).device
        N, D, Z = self._N, self._D, self._Z
        return TokenRSSMState(
            deter=torch.zeros(batch_size, N, D, device=device),
            stoch=torch.zeros(batch_size, N, Z, device=device),
            mean=torch.zeros(batch_size, Z, device=device),
            std=torch.ones(batch_size, Z, device=device),
            token_mean=torch.zeros(batch_size, N, Z, device=device),
            token_std=torch.ones(batch_size, N, Z, device=device),
        )

    def get_feat(self, state: TokenRSSMState) -> Tensor:
        """Pool token features to (B, feat_dim=320) for legacy interfaces.

        Uses mean+max concatenation to preserve localized token evidence
        (e.g., a single obstacle token) that mean pooling would dilute.
        Only called by EFE scorer, iCEM, and state/image decoders that
        expect the legacy (B, 320) interface.
        """
        tok = torch.cat([state.deter, state.stoch], dim=-1)  # (B, N, 320)
        mean_p = tok.mean(dim=1)                              # (B, 320)
        max_p = tok.max(dim=1).values                         # (B, 320)
        return self._feat_adapter(
            torch.cat([mean_p, max_p], dim=-1)                # (B, 640)
        )                                                      # (B, 320)

    def get_dist(self, state: TokenRSSMState) -> Normal:
        return Normal(state.mean, state.std)

    def img_step(self, prev_state: TokenRSSMState, prev_action: Tensor) -> TokenRSSMState:
        """Prior transition: propagate token dynamics without observations."""
        # prev_state.deter: (B, N, D)  prev_state.stoch: (B, N, Z)
        x = self._in_proj(
            torch.cat([prev_state.deter, prev_state.stoch], dim=-1)
        )                                                          # (B, N, D)

        # A1: Warp token grid by action flow BEFORE positional embedding.
        # This spatially displaces tokens according to the predicted ego motion,
        # so the prior "looks ahead" in image space before self-attention.
        if self._use_action_warp:
            x = self._apply_warp(x, prev_action)

        x = x + self._pos_embed                                    # (B, N, D)

        # Broadcast action embedding to every token.
        act_emb = self._action_mlp(prev_action)                    # (B, D)
        x = x + act_emb.unsqueeze(1)                               # (B, N, D)

        # Prior transformer: new deterministic context.
        new_deter = self._prior_transformer(x)                     # (B, N, D)

        # Stochastic prior head.
        raw = self._prior_head(new_deter)                          # (B, N, Z*2)
        token_mean, token_std, stoch_new = self._stoch_from_raw(raw)

        return self._make_state(new_deter, token_mean, token_std, stoch_new)

    def obs_step(
        self,
        prev_state: TokenRSSMState,
        prev_action: Tensor,
        embed: Tensor,
    ) -> tuple[TokenRSSMState, TokenRSSMState]:
        """Posterior update: fuse prior transition with observation tokens.

        embed: (B, N, embed_dim) from TokenViTEncoder.
        Returns (posterior, prior).
        """
        prior = self.img_step(prev_state, prev_action)

        # prior.deter is already (B, N, D) — no reshape needed.
        fused = self._obs_fuse(
            torch.cat([prior.deter, embed], dim=-1)               # (B, N, D+E)
        )                                                           # (B, N, D)
        fused = fused + self._pos_embed                            # (B, N, D)

        # Posterior transformer.
        post_x = self._post_transformer(fused)                     # (B, N, D)

        # Posterior stochastic head.
        raw = self._post_head(post_x)                              # (B, N, Z*2)
        token_mean, token_std, stoch_post = self._stoch_from_raw(raw)

        # Posterior shares prior's deter (RSSM convention).
        post = self._make_state(prior.deter, token_mean, token_std, stoch_post)
        return post, prior

    def imagine(self, initial_state: TokenRSSMState, actions: Tensor) -> list[TokenRSSMState]:
        """Open-loop rollout under action sequence.

        actions: (H, B, action_dim)
        Returns list of H TokenRSSMState objects.
        """
        states: list[TokenRSSMState] = []
        state = initial_state
        for t in range(actions.shape[0]):
            state = self.img_step(state, actions[t])
            states.append(state)
        return states


# ---------------------------------------------------------------------------
# TokenImageDecoder
# ---------------------------------------------------------------------------

class TokenImageDecoder(nn.Module):
    """Decode token grid back to image: (B,N,D), (B,N,Z) -> (B,3,64,64).

    forward(deter, stoch) accepts 3D token tensors directly.
    decode_from_feat(feat) accepts pooled (B, feat_dim) from get_feat()
    for the EFE visual-surprise path.
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
        self._dec_dim = dec_dim
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

        # For decode_from_feat: project pooled (B, feat_dim) to dec_dim,
        # then broadcast to spatial grid for the deconv stack.
        self._feat_proj_1d = nn.Linear(feat_dim, dec_dim)

    def forward(self, deter: Tensor, stoch: Tensor) -> Tensor:
        # deter: (B, N, D_deter)  stoch: (B, N, Z)
        B = deter.shape[0]
        tok = torch.cat([deter, stoch], dim=-1)              # (B, N, 320)
        tok = self._tok_proj(tok)                             # (B, N, dec_dim)
        grid_side = int(self._num_tokens ** 0.5)             # 8
        grid = tok.permute(0, 2, 1).view(
            B, -1, grid_side, grid_side,
        )                                                     # (B, dec_dim, 8, 8)
        return self._deconv(grid)                             # (B, 3, 64, 64)

    def decode_from_feat(self, feat: Tensor) -> Tensor:
        """Decode from pooled feature vector — used by EFE visual surprise.

        feat: (B, feat_dim=320) from TokenViTTransition.get_feat()
        Returns: (B, 3, 64, 64)
        """
        B = feat.shape[0]
        grid_side = int(self._num_tokens ** 0.5)             # 8
        x = self._feat_proj_1d(feat)                         # (B, dec_dim)
        # Broadcast single feature to spatial grid.
        x = x.view(B, self._dec_dim, 1, 1).expand(
            -1, -1, grid_side, grid_side,
        )                                                     # (B, dec_dim, 8, 8)
        return self._deconv(x)                               # (B, 3, 64, 64)


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
