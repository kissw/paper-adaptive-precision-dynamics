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
) -> tuple[Tensor, dict[str, Tensor]]:
    post = Normal(post_mean, post_std)
    prior = Normal(prior_mean, prior_std)

    # Dual KL: dyn trains prior (post detached), rep trains posterior (prior detached).
    # .sum(-1) reduces the stochastic dim Z; .mean() then averages over all remaining
    # prefix dims (B for RSSM; B×N for token-level world models).  Both shapes work.
    dyn_loss = kl_divergence(Normal(post_mean.detach(), post_std.detach()), prior).sum(-1).mean()
    rep_loss = kl_divergence(post, Normal(prior_mean.detach(), prior_std.detach())).sum(-1).mean()

    dyn_loss = torch.clamp(dyn_loss, min=free_nats)
    rep_loss = torch.clamp(rep_loss, min=free_nats)

    # Image reconstruction: per-dimension MSE
    c, h, w = obs_img.shape[1], obs_img.shape[2], obs_img.shape[3]
    img_loss = F.mse_loss(recon_img, obs_img, reduction="none").sum(dim=(1, 2, 3)).mean() / (
        c * h * w
    )

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


def warp_smoothness_loss(delta: Tensor, grid_side: int) -> Tensor:
    """Total-variation penalty on a per-token 2D warp field.

    delta: (B, N, 2) where N = grid_side**2. Reshapes to (B, G, G, 2) and
    penalizes squared differences between spatially adjacent tokens so the
    learned flow stays locally smooth (standard optical-flow regularizer).
    """
    B = delta.shape[0]
    G = grid_side
    d = delta.reshape(B, G, G, 2)
    dx = (d[:, :, 1:, :] - d[:, :, :-1, :]).pow(2).mean()
    dy = (d[:, 1:, :, :] - d[:, :-1, :, :]).pow(2).mean()
    return dx + dy


def action_contrastive_loss(
    pred_feat: Tensor,
    target_feat: Tensor,
    cf_feats: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE: real-action prediction should match the true next-state
    feature and be distinguishable from counterfactual-action predictions.

    pred_feat   : (B, F)     feat of img_step(prev_state, real_action)
    target_feat : (B, F)     feat of the true next posterior (detached)
    cf_feats    : (B, K, F)  feats of img_step under K shuffled actions
    Returns scalar InfoNCE loss (lower = action better predicts its own future).
    """
    p = F.normalize(pred_feat, dim=-1)
    t = F.normalize(target_feat.detach(), dim=-1)
    cf = F.normalize(cf_feats, dim=-1)                       # (B,K,F)

    pos = (p * t).sum(-1, keepdim=True)                      # (B,1)
    neg = torch.bmm(cf, p.unsqueeze(-1)).squeeze(-1)         # (B,K)
    logits = torch.cat([pos, neg], dim=1) / temperature      # (B,1+K)
    labels = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
    return F.cross_entropy(logits, labels)
