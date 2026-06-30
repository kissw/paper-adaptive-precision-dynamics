import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from active_inference.utils.transforms import symlog


def compute_vfe(
    post_mean: Tensor,
    post_std: Tensor,
    prior_mean: Tensor,
    prior_std: Tensor,
    obs_img: Tensor,
    recon_img: Tensor,
    obs_state: Tensor,
    recon_state: Tensor,
    free_nats: float = 1.0,
    kl_dyn_scale: float = 1.0,
    kl_rep_scale: float = 0.1,
    token_kl_weighting: str = "none",
    obstacle_bbox: Tensor | None = None,
    img_recon_obstacle_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute Variational Free Energy.

    token_kl_weighting — A3 token-weighted dynamics KL:
      "none"  : standard mean (default, RSSM/ViT identical)
      "error" : weight token KL by its own magnitude (error-feedback weighting).
                Only active when post_mean is 3D (B, N, Z); falls back to "none"
                for 2D RSSM states.

    obstacle_bbox / img_recon_obstacle_weight — obstacle-region image upweighting:
      When img_recon_obstacle_weight > 1 and obstacle_bbox is given (B,4 pixel xyxy
      in model space), pixels inside each bbox contribute proportionally more to
      img_loss.  weight=1.0 or bbox=None → standard uniform MSE (regression-safe).
    """
    post = Normal(post_mean, post_std)
    prior = Normal(prior_mean, prior_std)

    # Dual KL: dyn trains prior (post detached), rep trains posterior (prior detached).
    # .sum(-1) reduces the stochastic dim Z; .mean() then averages over all remaining
    # prefix dims (B for RSSM; B×N for token-level world models).  Both shapes work.
    per_token_kl_dyn = kl_divergence(
        Normal(post_mean.detach(), post_std.detach()), prior
    ).sum(-1)  # (B,) for RSSM, (B, N) for ViT

    if token_kl_weighting == "error" and post_mean.ndim == 3:
        # Error-weighted: tokens with larger current KL get proportionally higher weight.
        # Normalise per-sample so total weight == N (preserves magnitude).
        w = per_token_kl_dyn.detach()
        w = w / (w.mean(dim=1, keepdim=True) + 1e-8)
        dyn_loss = (per_token_kl_dyn * w).mean()
    else:
        dyn_loss = per_token_kl_dyn.mean()

    rep_loss = kl_divergence(post, Normal(prior_mean.detach(), prior_std.detach())).sum(-1).mean()

    dyn_loss = torch.clamp(dyn_loss, min=free_nats)
    rep_loss = torch.clamp(rep_loss, min=free_nats)

    # Image reconstruction: per-dimension MSE
    c, h, w = obs_img.shape[1], obs_img.shape[2], obs_img.shape[3]
    err = F.mse_loss(recon_img, obs_img, reduction="none")  # (B,C,H,W)
    if img_recon_obstacle_weight > 1.0 and obstacle_bbox is not None:
        # Reuse the same bbox→weight-map helper as the rollout recon path.
        # Lazy import avoids a circular import (agent imports this module).
        from active_inference.agent import DeepAIFAgent
        w_map = DeepAIFAgent._bbox_weight_map(
            obstacle_bbox, recon_img.shape,
            img_recon_obstacle_weight, recon_img.device,
        )  # (B,1,H,W)
        err = err * w_map
    img_loss = err.sum(dim=(1, 2, 3)).mean() / (c * h * w)

    # State reconstruction with symlog
    state_loss = F.mse_loss(symlog(recon_state), symlog(obs_state))

    total = img_loss + state_loss + kl_dyn_scale * dyn_loss + kl_rep_scale * rep_loss

    # KL per dim for monitoring partial collapse
    kl_per_dim = kl_divergence(post, prior).mean(0)

    return total, {
        "img_loss": img_loss,
        "state_loss": state_loss,
        "kl_dyn": dyn_loss,
        "kl_rep": rep_loss,
        "kl_per_dim": kl_per_dim,
    }


def compute_overshoot_kl(
    prior_mean: Tensor,
    prior_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    free_nats: float = 1.0,
) -> Tensor:
    """KL(sg_posterior_target ‖ imagined_prior) — dynamics direction.

    Trains the multi-step open-loop prior toward the observed posterior,
    reducing accumulated error in long-horizon rollouts.

    target_* must be stop-gradient (called from the no-grad reference pass).
    """
    kl = kl_divergence(
        Normal(target_mean, target_std),
        Normal(prior_mean, prior_std),
    ).sum(-1).mean()
    return torch.clamp(kl, min=free_nats)


def compute_transition_target_kl(
    prior_mean: Tensor,
    prior_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return raw and free-nats-clamped KL(target posterior || online prior)."""
    kl_raw = kl_divergence(
        Normal(target_mean.detach(), target_std.detach()),
        Normal(prior_mean, prior_std),
    ).sum(-1).mean()
    return kl_raw, torch.clamp(kl_raw, min=free_nats)
