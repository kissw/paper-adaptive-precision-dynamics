import torch
from torch.distributions import Normal, kl_divergence

from active_inference.training.losses import compute_vfe


def test_vfe_known_analytical():
    post_mean = torch.tensor([[1.0, 2.0]])
    post_std = torch.tensor([[0.5, 1.0]])
    prior_mean = torch.tensor([[0.0, 0.0]])
    prior_std = torch.tensor([[1.0, 1.0]])
    expected_kl = (
        kl_divergence(Normal(post_mean, post_std), Normal(prior_mean, prior_std)).sum(-1).mean()
    )
    img = torch.zeros(1, 3, 64, 64)
    st = torch.zeros(1, 2)
    _, info = compute_vfe(post_mean, post_std, prior_mean, prior_std, img, img, st, st)
    assert abs(info["kl_dyn"].item() - max(expected_kl.item(), 1.0)) < 1e-4
    assert abs(info["kl_rep"].item() - max(expected_kl.item(), 1.0)) < 1e-4


def test_stop_gradient_dyn():
    post_mean = torch.randn(4, 8, requires_grad=True)
    post_std = torch.ones(4, 8) * 0.5
    prior_mean = torch.randn(4, 8, requires_grad=True)
    prior_std = torch.ones(4, 8)
    img = torch.zeros(4, 3, 64, 64)
    st = torch.zeros(4, 2)
    _, info = compute_vfe(post_mean, post_std, prior_mean, prior_std, img, img, st, st)
    grad = torch.autograd.grad(info["kl_dyn"], post_mean, retain_graph=True, allow_unused=True)[0]
    assert grad is None


def test_stop_gradient_rep():
    post_mean = torch.randn(4, 8, requires_grad=True)
    post_std = torch.ones(4, 8) * 0.5
    prior_mean = torch.randn(4, 8, requires_grad=True)
    prior_std = torch.ones(4, 8)
    img = torch.zeros(4, 3, 64, 64)
    st = torch.zeros(4, 2)
    _, info = compute_vfe(post_mean, post_std, prior_mean, prior_std, img, img, st, st)
    grad = torch.autograd.grad(info["kl_rep"], prior_mean, retain_graph=True, allow_unused=True)[0]
    assert grad is None


def test_free_nats_clipping():
    m = torch.zeros(4, 8)
    s = torch.ones(4, 8)
    img = torch.zeros(4, 3, 64, 64)
    st = torch.zeros(4, 2)
    _, info = compute_vfe(m, s, m, s, img, img, st, st, free_nats=1.0)
    assert info["kl_dyn"].item() >= 1.0 - 1e-6
    assert info["kl_rep"].item() >= 1.0 - 1e-6


def test_loss_scale_balance():
    post_mean = torch.randn(8, 32)
    post_std = torch.ones(8, 32) * 0.5
    prior_mean = torch.randn(8, 32)
    prior_std = torch.ones(8, 32)
    img = torch.randn(8, 3, 64, 64)
    recon = img + torch.randn_like(img) * 0.5
    st = torch.randn(8, 2)
    _, info = compute_vfe(post_mean, post_std, prior_mean, prior_std, img, recon, st, st)
    assert info["img_loss"].item() > 0
    assert info["kl_dyn"].item() > 0


def test_vfe_no_nan_with_small_std():
    m = torch.randn(4, 64)
    s = torch.ones(4, 64) * 0.1
    img = torch.randn(4, 3, 64, 64)
    st = torch.randn(4, 2)
    total, _ = compute_vfe(m, s, m, s, img, img, st, st)
    assert not torch.isnan(total)


def test_kl_per_dim_shape():
    m = torch.randn(4, 16)
    s = torch.ones(4, 16)
    img = torch.zeros(4, 3, 64, 64)
    st = torch.zeros(4, 2)
    _, info = compute_vfe(m, s, m, s, img, img, st, st)
    assert info["kl_per_dim"].shape == (16,)
