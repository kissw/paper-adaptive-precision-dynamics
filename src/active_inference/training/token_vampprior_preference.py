"""Token VampPrior-like preference model for TokenViT.

Unlike the Shared/Position-normalised Token GMM (which operates on
concat(deter,stoch) 320D features), this module works directly with the
token posterior distribution (token_mean, token_std) — 64D per token.

Key idea:
  Each spatial position p has its own K-component Gaussian mixture fitted
  on clean/avoid token posterior statistics (posterior centres and spreads).
  At inference time the preference score integrates over the current token
  posterior q(z) = Normal(token_mean, token_std), yielding:

      E_q[log N(z; mu_k, sigma_k^2)] = log N(q_mean; mu_k, sqrt(sigma_k^2 + q_std^2))

  which is the "q-integrated" mixture density.  This is a VampPrior-like
  construction: the pseudo-inputs are the posterior statistics of the
  clean/avoid reference episodes.

Score per token:
    s(q_n) = log p_clean(q_n | pos=n) - contrast_scale * log p_avoid(q_n | pos=n)

Per-frame aggregation uses top-k mean over N token scores.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Posterior extraction
# ---------------------------------------------------------------------------

def extract_token_posterior_stats(state) -> tuple[Tensor, Tensor]:
    """Extract token_mean / token_std from a TokenRSSMState.

    Returns:
        token_mean : (B, N, Z)
        token_std  : (B, N, Z)

    Raises ValueError for RSSM 2D states or missing token fields.
    """
    if not hasattr(state, "token_mean") or not hasattr(state, "token_std"):
        raise ValueError(
            "Token VampPrior-like preference requires TokenViT state with "
            "token_mean / token_std fields. Got state type: "
            f"{type(state).__name__}. Use pooled GMM for RSSM states."
        )
    tm = state.token_mean
    ts = state.token_std
    if tm.ndim != 3 or ts.ndim != 3:
        raise ValueError(
            "Token VampPrior-like preference requires token_mean/token_std with "
            f"ndim=3 (B,N,Z). Got token_mean.ndim={tm.ndim}, "
            f"token_std.ndim={ts.ndim}. Use pooled GMM for RSSM 2D states."
        )
    return tm, ts


# ---------------------------------------------------------------------------
# KMeans-based posterior mixture fitting (per position)
# ---------------------------------------------------------------------------

def fit_position_posterior_mixture(
    token_mean: Tensor,
    token_std: Tensor,
    K: int = 5,
    seed: int = 42,
    min_std: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a K-component Gaussian mixture per spatial position.

    Uses KMeans on posterior means to choose component centres, then
    computes component variance as the posterior-aware mixture variance:

        var_k = mean_{i in cluster_k}(std_i^2 + (mean_i - centre_k)^2)

    This correctly accounts for the spread of posterior distributions
    assigned to each component (cf. mixture-of-posteriors).

    Args:
        token_mean : (N_frames, N_tokens, Z)
        token_std  : (N_frames, N_tokens, Z)
        K          : number of mixture components
        seed       : random seed for KMeans
        min_std    : minimum std after clamping

    Returns:
        proto_mean    : (N_tokens, K, Z)  component means
        proto_log_std : (N_tokens, K, Z)  log of component stds
        proto_logits  : (N_tokens, K)     log(cluster_count / total)
    """
    from sklearn.cluster import KMeans  # soft dependency, available in project

    N_frames, N_pos, Z = token_mean.shape

    proto_mean    = torch.zeros(N_pos, K, Z)
    proto_log_std = torch.zeros(N_pos, K, Z)
    proto_logits  = torch.zeros(N_pos, K)

    mu_np  = token_mean.cpu().numpy()   # (F, N, Z)
    std_np = token_std.cpu().numpy()    # (F, N, Z)

    for p in range(N_pos):
        mu_p  = mu_np[:, p, :]   # (F, Z)
        std_p = std_np[:, p, :]  # (F, Z)

        # KMeans on posterior means for this position.
        km = KMeans(n_clusters=K, random_state=seed, n_init=5)
        labels = km.fit_predict(mu_p)   # (F,)
        centres = km.cluster_centers_   # (K, Z)

        for k in range(K):
            mask = labels == k
            if mask.sum() == 0:
                # Empty cluster: fall back to nearest prototype from the full set
                dists = np.linalg.norm(mu_p - centres[k], axis=-1)
                nearest = np.argmin(dists)
                proto_mean[p, k]    = torch.tensor(mu_p[nearest], dtype=torch.float32)
                proto_log_std[p, k] = math.log(min_std)
                proto_logits[p, k]  = math.log(1.0 / (N_frames + 1))
            else:
                mu_k  = mu_p[mask]    # (M, Z)
                std_k = std_p[mask]   # (M, Z)
                centre_k = centres[k] # (Z,)

                # Mixture component variance: E[var_i] + E[(mean_i - centre)^2]
                var_k = (std_k ** 2 + (mu_k - centre_k) ** 2).mean(axis=0)  # (Z,)
                std_k_out = np.sqrt(np.maximum(var_k, min_std ** 2))

                proto_mean[p, k]    = torch.tensor(centre_k, dtype=torch.float32)
                proto_log_std[p, k] = torch.tensor(
                    np.log(std_k_out), dtype=torch.float32,
                )
                count = mask.sum()
                proto_logits[p, k] = math.log(count / N_frames)

    return proto_mean, proto_log_std, proto_logits


# ---------------------------------------------------------------------------
# Q-integrated mixture log probability
# ---------------------------------------------------------------------------

def log_prob_posterior_under_mixture(
    q_mean: Tensor,
    q_std: Tensor,
    proto_mean: Tensor,
    proto_log_std: Tensor,
    proto_logits: Tensor,
    q_std_scale: float = 1.0,
    proto_std_scale: float = 1.0,
    min_var: float = 1e-12,
) -> Tensor:
    """Compute E_q[log p_mix(z)] using the q-integrated density approximation.

    The integral of q(z) N(z; mu_k, sigma_k^2) dz equals
    N(q_mean; mu_k, sqrt(sigma_k^2 + q_std^2)), so:

        log p(q) = logsumexp_k( log_weight_k + log N(q_mean; mu_k, eff_std_k) )

    Scaled variant:
        effective_var = (proto_std_scale * proto_std)^2 + (q_std_scale * q_std)^2

    q_std_scale=1.0, proto_std_scale=1.0 reproduces the original formula.
    q_std_scale=0.0 ignores posterior uncertainty (approaches mean_only).
    proto_std_scale<1.0 sharpens the prototype mixture.

    Args:
        q_mean          : (B, N, Z)
        q_std           : (B, N, Z)
        proto_mean      : (N, K, Z)
        proto_log_std   : (N, K, Z)
        proto_logits    : (N, K)
        q_std_scale     : scale applied to q_std before squaring (default 1.0)
        proto_std_scale : scale applied to proto_std before squaring (default 1.0)
        min_var         : minimum effective variance after clamping

    Returns:
        log_prob : (B, N)
    """
    # Broadcast: q_mean/q_std (B,N,1,Z), proto (1,N,K,Z)
    qm  = q_mean.unsqueeze(2)                       # (B, N, 1, Z)
    qs  = q_std.unsqueeze(2)                        # (B, N, 1, Z)
    pm  = proto_mean.unsqueeze(0)                   # (1, N, K, Z)
    pls = proto_log_std.unsqueeze(0)                # (1, N, K, Z)

    proto_var   = (2 * pls).exp()                   # (1, N, K, Z)  sigma_k^2
    eff_var     = (
        (proto_std_scale ** 2) * proto_var
        + (q_std_scale ** 2) * qs ** 2
    ).clamp(min=min_var)                            # (B, N, K, Z)
    eff_std     = eff_var.sqrt()                    # (B, N, K, Z)

    # log N(q_mean; proto_mean, eff_std)
    diff        = qm - pm                          # (B, N, K, Z)
    log_gauss   = (
        -0.5 * math.log(2 * math.pi)
        - eff_std.log()
        - 0.5 * (diff / eff_std) ** 2
    ).sum(-1)                                       # (B, N, K)

    # Normalise logits to log-weights
    log_weights = proto_logits.unsqueeze(0) - torch.logsumexp(
        proto_logits, dim=-1, keepdim=True,
    ).unsqueeze(0)                                  # (1, N, K)

    return torch.logsumexp(log_weights + log_gauss, dim=-1)   # (B, N)


def log_prob_mean_only(
    q_mean: Tensor,
    proto_mean: Tensor,
    proto_log_std: Tensor,
    proto_logits: Tensor,
    proto_std_scale: float = 1.0,
    min_var: float = 1e-12,
) -> Tensor:
    """Ablation: evaluate mixture at q_mean only (ignores q_std).

    proto_std_scale controls prototype sharpness; q_std is ignored.

    Args:
        q_mean          : (B, N, Z)
        proto_mean      : (N, K, Z)
        proto_log_std   : (N, K, Z)
        proto_logits    : (N, K)
        proto_std_scale : scale applied to proto_std (default 1.0)
        min_var         : minimum variance after clamping

    Returns:
        log_prob : (B, N)
    """
    qm  = q_mean.unsqueeze(2)            # (B, N, 1, Z)
    pm  = proto_mean.unsqueeze(0)        # (1, N, K, Z)
    pls = proto_log_std.unsqueeze(0)     # (1, N, K, Z)

    # Scaled proto variance
    proto_var = (2 * pls).exp()          # (1, N, K, Z)
    scaled_var = ((proto_std_scale ** 2) * proto_var).clamp(min=min_var)
    scaled_std = scaled_var.sqrt()       # (1, N, K, Z)

    diff      = qm - pm                  # (B, N, K, Z)
    log_gauss = (
        -0.5 * math.log(2 * math.pi)
        - scaled_std.log()
        - 0.5 * (diff / scaled_std) ** 2
    ).sum(-1)                            # (B, N, K)

    log_weights = proto_logits.unsqueeze(0) - torch.logsumexp(
        proto_logits, dim=-1, keepdim=True,
    ).unsqueeze(0)                       # (1, N, K)

    return torch.logsumexp(log_weights + log_gauss, dim=-1)  # (B, N)


# ---------------------------------------------------------------------------
# Top-k aggregation
# ---------------------------------------------------------------------------

def topk_mean_score(token_scores: Tensor, k: int) -> Tensor:
    """Aggregate per-token scores using top-k mean.

    token_scores : (B, N)
    Returns      : (B,)  — k=1 is max-like, k=N is mean-like.
    """
    k = min(k, token_scores.shape[-1])
    topk_vals, _ = torch.topk(token_scores, k=k, dim=-1)
    return topk_vals.mean(dim=-1)


# ---------------------------------------------------------------------------
# TokenVampPriorPreference
# ---------------------------------------------------------------------------

class TokenVampPriorPreference:
    """Position-conditional posterior mixture preference for TokenViT.

    Stores per-position K-component Gaussian mixtures for clean and
    obstacle-avoidance references.  Scoring integrates over the current
    token posterior q(z) = Normal(token_mean, token_std).
    """

    def __init__(
        self,
        proto_mean_clean: Tensor,     # (N, K_clean, Z)
        proto_log_std_clean: Tensor,  # (N, K_clean, Z)
        proto_logits_clean: Tensor,   # (N, K_clean)
        proto_mean_avoid: Tensor,     # (N, K_avoid, Z)
        proto_log_std_avoid: Tensor,  # (N, K_avoid, Z)
        proto_logits_avoid: Tensor,   # (N, K_avoid)
        contrast_scale: float = 1.0,
        score_mode: str = "q_integrated",
        topk_default: int = 4,
        topk_candidates: list[int] | None = None,
        q_std_scale: float = 1.0,
        proto_std_scale: float = 1.0,
    ):
        self.clean_mean    = proto_mean_clean
        self.clean_log_std = proto_log_std_clean
        self.clean_logits  = proto_logits_clean
        self.avoid_mean    = proto_mean_avoid
        self.avoid_log_std = proto_log_std_avoid
        self.avoid_logits  = proto_logits_avoid
        self.contrast_scale    = contrast_scale
        self.score_mode        = score_mode
        self.topk_default      = topk_default
        self.topk_candidates   = topk_candidates or [1, 4, 8, 16]
        self.q_std_scale       = q_std_scale
        self.proto_std_scale   = proto_std_scale

    def to(self, device):
        self.clean_mean    = self.clean_mean.to(device)
        self.clean_log_std = self.clean_log_std.to(device)
        self.clean_logits  = self.clean_logits.to(device)
        self.avoid_mean    = self.avoid_mean.to(device)
        self.avoid_log_std = self.avoid_log_std.to(device)
        self.avoid_logits  = self.avoid_logits.to(device)
        return self

    def _log_prob_clean(self, q_mean: Tensor, q_std: Tensor) -> Tensor:
        if self.score_mode == "q_integrated":
            return log_prob_posterior_under_mixture(
                q_mean, q_std, self.clean_mean, self.clean_log_std, self.clean_logits,
                q_std_scale=self.q_std_scale, proto_std_scale=self.proto_std_scale,
            )
        return log_prob_mean_only(
            q_mean, self.clean_mean, self.clean_log_std, self.clean_logits,
            proto_std_scale=self.proto_std_scale,
        )

    def _log_prob_avoid(self, q_mean: Tensor, q_std: Tensor) -> Tensor:
        if self.score_mode == "q_integrated":
            return log_prob_posterior_under_mixture(
                q_mean, q_std, self.avoid_mean, self.avoid_log_std, self.avoid_logits,
                q_std_scale=self.q_std_scale, proto_std_scale=self.proto_std_scale,
            )
        return log_prob_mean_only(
            q_mean, self.avoid_mean, self.avoid_log_std, self.avoid_logits,
            proto_std_scale=self.proto_std_scale,
        )

    def token_scores(self, q_mean: Tensor, q_std: Tensor) -> Tensor:
        """Contrastive score per token.

        Args:
            q_mean : (B, N, Z)
            q_std  : (B, N, Z)

        Returns:
            scores : (B, N)  positive = clean-like, negative = obstacle-like
        """
        lp_clean = self._log_prob_clean(q_mean, q_std)  # (B, N)
        lp_avoid = self._log_prob_avoid(q_mean, q_std)  # (B, N)
        return lp_clean - self.contrast_scale * lp_avoid

    def score_frames(
        self,
        q_mean: Tensor,
        q_std: Tensor,
        topk: int | None = None,
        direction: str = "clean",
    ) -> Tensor:
        """Top-k mean score per frame.

        Args:
            q_mean    : (B, N, Z)
            q_std     : (B, N, Z)
            topk      : number of top tokens; defaults to topk_default
            direction : "clean"    — top-k largest clean-preference score
                        "obstacle" — top-k largest obstacle energy (= -score)

        Returns:
            (B,)
        """
        if topk is None:
            topk = self.topk_default
        scores = self.token_scores(q_mean, q_std)  # (B, N)
        if direction == "obstacle":
            scores = -scores
        return topk_mean_score(scores, k=topk)

    def state_dict(self) -> dict:
        return {
            "clean_mean":    self.clean_mean.cpu(),
            "clean_log_std": self.clean_log_std.cpu(),
            "clean_logits":  self.clean_logits.cpu(),
            "avoid_mean":    self.avoid_mean.cpu(),
            "avoid_log_std": self.avoid_log_std.cpu(),
            "avoid_logits":  self.avoid_logits.cpu(),
            "contrast_scale":  self.contrast_scale,
            "score_mode":      self.score_mode,
            "topk_default":    self.topk_default,
            "topk_candidates": self.topk_candidates,
            "q_std_scale":     self.q_std_scale,
            "proto_std_scale": self.proto_std_scale,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "TokenVampPriorPreference":
        return cls(
            proto_mean_clean=sd["clean_mean"],
            proto_log_std_clean=sd["clean_log_std"],
            proto_logits_clean=sd["clean_logits"],
            proto_mean_avoid=sd["avoid_mean"],
            proto_log_std_avoid=sd["avoid_log_std"],
            proto_logits_avoid=sd["avoid_logits"],
            contrast_scale=sd.get("contrast_scale", 1.0),
            score_mode=sd.get("score_mode", "q_integrated"),
            topk_default=sd.get("topk_default", 4),
            topk_candidates=sd.get("topk_candidates", [1, 4, 8, 16]),
            q_std_scale=sd.get("q_std_scale", 1.0),
            proto_std_scale=sd.get("proto_std_scale", 1.0),
        )
