"""Token-wise contrastive preference model for TokenViT world model.

Shared Token GMM: fits two PreferenceModels on flattened token features
(M, token_dim) where M = B * N, capturing per-token distribution.

Two modes:
  shared   — raw token features, no position correction
  pos_norm — position-normalised features:
               mu_pos, std_pos computed from clean tokens only;
               same stats applied to both clean and obstacle tokens before GMM

Contrastive score per token:
    score(z_n) = log p_clean(z_n) - contrast_scale * log p_avoid(z_n)

Per-frame aggregation uses top-k mean over N token scores.
k=1 → max-like (most discriminative token), k=N → mean-like.

Feature extraction uses concat(deter, stoch) by default (deter_stoch),
not get_feat() or pooled compatibility fields.
"""

from __future__ import annotations

import torch
from torch import Tensor

from active_inference.training.preference import PreferenceModel


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_token_features(state, feature_type: str = "deter_stoch") -> Tensor:
    """Extract per-token feature tensor from a TokenRSSMState.

    state.deter must be 3D (B, N, D). Raises ValueError for RSSM 2D states.

    feature_type choices and output dims:
      deter_stoch      -> cat(deter, stoch)       (B, N, D+Z=320)
      deter_token_mean -> cat(deter, token_mean)  (B, N, D+Z=320)
      token_mean       -> token_mean              (B, N,   Z=64)
    """
    if state.deter.ndim != 3:
        raise ValueError(
            "Token-wise contrastive preference requires TokenViT state with 3D "
            f"deter/stoch. Got deter.ndim={state.deter.ndim} (expected 3). "
            "Use pooled contrastive preference (fit_contrastive_preference.py) "
            "for RSSM 2D states."
        )
    if feature_type == "deter_stoch":
        return torch.cat([state.deter, state.stoch], dim=-1)        # (B, N, D+Z)
    elif feature_type == "deter_token_mean":
        return torch.cat([state.deter, state.token_mean], dim=-1)   # (B, N, D+Z)
    elif feature_type == "token_mean":
        return state.token_mean                                      # (B, N, Z)
    else:
        raise ValueError(
            f"Unknown feature_type: {feature_type!r}. "
            "Valid choices: deter_stoch, deter_token_mean, token_mean"
        )


# ---------------------------------------------------------------------------
# Position normalization
# ---------------------------------------------------------------------------

def compute_position_stats(
    clean_tokens: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute per-position mean and std from clean token frames.

    Args:
        clean_tokens: (N_frames, N_tokens, token_dim)
        eps: minimum std after clamping

    Returns:
        mu_pos:  (N_tokens, token_dim) — mean over frames
        std_pos: (N_tokens, token_dim) — std over frames, clamped >= eps
    """
    mu_pos  = clean_tokens.mean(dim=0)               # (N, D)
    std_pos = clean_tokens.std(dim=0).clamp(min=eps)  # (N, D)
    return mu_pos, std_pos


def apply_position_normalization(
    tokens: Tensor,
    mu_pos: Tensor,
    std_pos: Tensor,
) -> Tensor:
    """Normalise tokens per spatial position.

    Args:
        tokens:  (N_frames, N_tokens, token_dim)
        mu_pos:  (N_tokens, token_dim)
        std_pos: (N_tokens, token_dim)

    Returns:
        (N_frames, N_tokens, token_dim)  zero-mean unit-std per position.
    """
    return (tokens - mu_pos.unsqueeze(0)) / std_pos.unsqueeze(0)


# ---------------------------------------------------------------------------
# Top-k aggregation
# ---------------------------------------------------------------------------

def topk_mean_token_score(token_scores: Tensor, k: int) -> Tensor:
    """Aggregate per-token scores using top-k mean.

    token_scores: (B, N)
    Returns (B,) — k=1 is max-like, k=N is mean-like.
    """
    k = min(k, token_scores.shape[-1])
    topk_vals, _ = torch.topk(token_scores, k=k, dim=-1)  # (B, k)
    return topk_vals.mean(dim=-1)                          # (B,)


# ---------------------------------------------------------------------------
# TokenContrastivePreference
# ---------------------------------------------------------------------------

class TokenContrastivePreference:
    """Shared Token GMM contrastive preference for TokenViT.

    Both GMMs are standard PreferenceModel instances operating on flattened
    token features of shape (M, token_dim) where M = B * N.
    """

    def __init__(
        self,
        K_clean: int = 5,
        K_avoid: int = 5,
        token_dim: int = 320,
        min_std: float = 0.01,
        contrast_scale: float = 1.0,
    ):
        self.clean = PreferenceModel(K=K_clean, latent_dim=token_dim, min_std=min_std)
        self.avoid = PreferenceModel(K=K_avoid, latent_dim=token_dim, min_std=min_std)
        self._contrast_scale = contrast_scale
        self._token_dim = token_dim

    def to(self, device):
        self.clean = self.clean.to(device)
        self.avoid = self.avoid.to(device)
        return self

    def log_prob_flat(self, z_flat: Tensor) -> Tensor:
        """Contrastive log-ratio for flattened (M, token_dim) input."""
        return (
            self.clean.log_prob(z_flat)
            - self._contrast_scale * self.avoid.log_prob(z_flat)
        )  # (M,)

    def token_scores(self, z_tokens: Tensor) -> Tensor:
        """Per-token scores without aggregation. Returns (B, N)."""
        B, N, D = z_tokens.shape
        return self.log_prob_flat(z_tokens.reshape(B * N, D)).reshape(B, N)

    def score_frames(self, z_tokens: Tensor, topk: int = 4) -> Tensor:
        """Top-k mean score per frame. z_tokens: (B, N, token_dim) -> (B,)."""
        return topk_mean_token_score(self.token_scores(z_tokens), k=topk)

    def fit(
        self,
        clean_tokens_flat: Tensor,
        avoid_tokens_flat: Tensor,
        n_iters: int = 300,
        lr: float = 0.01,
    ):
        """Fit both GMMs on pre-flattened (M, token_dim) token samples."""
        print(
            f"Fitting clean token GMM  "
            f"(K={self.clean._K}, dim={self._token_dim}) "
            f"on {clean_tokens_flat.shape[0]} token samples..."
        )
        self.clean.update_from_latents(clean_tokens_flat, n_iters, lr)

        print(
            f"Fitting avoid token GMM  "
            f"(K={self.avoid._K}, dim={self._token_dim}) "
            f"on {avoid_tokens_flat.shape[0]} token samples..."
        )
        self.avoid.update_from_latents(avoid_tokens_flat, n_iters, lr)

        with torch.no_grad():
            c_score = self.log_prob_flat(clean_tokens_flat).mean().item()
            a_score = self.log_prob_flat(avoid_tokens_flat).mean().item()
            print(f"\nToken contrastive diagnostic (per-token):")
            print(f"  Clean tokens avg log-ratio: {c_score:+.2f}  (should be > 0)")
            print(f"  Avoid tokens avg log-ratio: {a_score:+.2f}  (should be < 0)")
            print(f"  Separation gap:             {c_score - a_score:.2f} nats")
