import torch
from active_inference.models import RSSM


def test_rssm_initial_state():
    rssm = RSSM()
    s = rssm.initial(4)
    assert s.deter.shape == (4, 256)
    assert s.stoch.shape == (4, 64)
    assert torch.all(s.deter == 0)
    assert torch.all(s.stoch == 0)


def test_rssm_hidden_state_changes():
    rssm = RSSM()
    s = rssm.initial(2)
    a = torch.randn(2, 2)
    e = torch.randn(2, 256)
    for _ in range(5):
        s, _ = rssm.obs_step(s, a, e)
    assert not torch.allclose(s.deter, torch.zeros_like(s.deter))


def test_obs_step_differs_from_img_step():
    rssm = RSSM()
    s = rssm.initial(4)
    a = torch.randn(4, 2)
    e = torch.randn(4, 256)
    post, prior = rssm.obs_step(s, a, e)
    assert not torch.allclose(post.mean, prior.mean, atol=1e-3)


def test_imagine_trajectory_shapes():
    rssm = RSSM()
    s = rssm.initial(4)
    actions = torch.randn(10, 4, 2)
    trajectory = rssm.imagine(s, actions)
    assert len(trajectory) == 10
    assert trajectory[0].deter.shape == (4, 256)
    assert trajectory[0].stoch.shape == (4, 64)


def test_get_feat_shape():
    rssm = RSSM()
    s = rssm.initial(4)
    feat = rssm.get_feat(s)
    assert feat.shape == (4, 320)


def test_rssm_gru_input_composition():
    rssm = RSSM()
    s = rssm.initial(2)
    a1 = torch.zeros(2, 2)
    a2 = torch.ones(2, 2)
    s1 = rssm.img_step(s, a1)
    s2 = rssm.img_step(s, a2)
    assert not torch.allclose(s1.deter, s2.deter)


def test_rssm_gradient_flows():
    rssm = RSSM()
    s = rssm.initial(2)
    a = torch.randn(2, 2)
    e = torch.randn(2, 256)
    post, _ = rssm.obs_step(s, a, e)
    loss = post.stoch.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in rssm.parameters())
    assert has_grad
