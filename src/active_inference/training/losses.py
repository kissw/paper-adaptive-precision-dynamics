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

    # Dual KL: dyn trains prior (post detached), rep trains posterior (prior detached)
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
