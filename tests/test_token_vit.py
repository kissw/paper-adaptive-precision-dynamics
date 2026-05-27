"""Unit tests for the token-level ViT world model.

Tests 1-8 verify shapes of individual components.
Test 9 verifies loss backward-pass compatibility.
Test 10 verifies RSSM regression (default config still selects RSSM).
"""

import torch
import pytest
from active_inference.config import Config
from active_inference.models.rssm import RSSMState
from active_inference.models.token_vit import (
    TokenViTEncoder,
    TokenViTTransition,
    TokenImageDecoder,
    TokenStateDecoder,
)
from active_inference.training.losses import compute_vfe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, N, D, Z = 2, 64, 256, 64  # batch, num_tokens, deter_dim, stoch_dim


def _encoder():
    return TokenViTEncoder(
        image_size=64, patch_size=8, embed_dim=256,
        state_dim=4, num_layers=4, num_heads=8,
    )


def _transition():
    return TokenViTTransition(
        num_tokens=N, embed_dim=D, deter_dim=D, stoch_dim=Z,
        action_dim=2, num_prior_layers=2, num_post_layers=2,
        num_heads=8,
    )


def _image_decoder():
    return TokenImageDecoder(num_tokens=N, deter_dim=D, stoch_dim=Z)


def _state_decoder():
    return TokenStateDecoder(feat_dim=320, state_dim=4)


# ---------------------------------------------------------------------------
# Test 1: Encoder output shape
# ---------------------------------------------------------------------------

def test_encoder_shape():
    enc = _encoder()
    img = torch.randn(B, 3, 64, 64)
    state = torch.randn(B, 4)
    out = enc(img, state)
    assert out.shape == (B, N, D), f"expected ({B},{N},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Initial state shapes
# ---------------------------------------------------------------------------

def test_initial_state_shapes():
    tr = _transition()
    s = tr.initial(B, device=torch.device("cpu"))
    assert s.deter.shape == (B, N * D),   f"deter: {s.deter.shape}"
    assert s.stoch.shape == (B, N * Z),   f"stoch: {s.stoch.shape}"
    assert s.mean.shape  == (B, Z),       f"mean:  {s.mean.shape}"
    assert s.std.shape   == (B, Z),       f"std:   {s.std.shape}"


# ---------------------------------------------------------------------------
# Test 3: img_step (prior) shapes
# ---------------------------------------------------------------------------

def test_img_step_shape():
    tr = _transition()
    s0 = tr.initial(B, device=torch.device("cpu"))
    action = torch.randn(B, 2)
    prior = tr.img_step(s0, action)
    assert prior.deter.shape == (B, N * D)
    assert prior.stoch.shape == (B, N * Z)
    assert prior.mean.shape  == (B, Z)
    assert prior.std.shape   == (B, Z)
    # std must be strictly positive
    assert (prior.std > 0).all()


# ---------------------------------------------------------------------------
# Test 4: obs_step returns (post, prior) with correct shapes
# ---------------------------------------------------------------------------

def test_obs_step_shape():
    enc = _encoder()
    tr = _transition()
    s0 = tr.initial(B, device=torch.device("cpu"))
    action = torch.randn(B, 2)
    img = torch.randn(B, 3, 64, 64)
    state = torch.randn(B, 4)
    obs_tokens = enc(img, state)              # (B, N, D)
    post, prior = tr.obs_step(s0, action, obs_tokens)
    for state_obj, label in [(post, "post"), (prior, "prior")]:
        assert state_obj.deter.shape == (B, N * D), f"{label}.deter"
        assert state_obj.stoch.shape == (B, N * Z), f"{label}.stoch"
        assert state_obj.mean.shape  == (B, Z),     f"{label}.mean"
        assert state_obj.std.shape   == (B, Z),     f"{label}.std"


# ---------------------------------------------------------------------------
# Test 5: imagine returns list of H states
# ---------------------------------------------------------------------------

def test_imagine_shape():
    tr = _transition()
    H = 5
    s0 = tr.initial(B, device=torch.device("cpu"))
    actions = torch.randn(H, B, 2)
    traj = tr.imagine(s0, actions)
    assert len(traj) == H
    for t, s in enumerate(traj):
        assert s.deter.shape == (B, N * D), f"step {t} deter"
        assert s.stoch.shape == (B, N * Z), f"step {t} stoch"


# ---------------------------------------------------------------------------
# Test 6: get_feat returns (B, 320)
# ---------------------------------------------------------------------------

def test_get_feat_shape():
    tr = _transition()
    s0 = tr.initial(B, device=torch.device("cpu"))
    feat = tr.get_feat(s0)
    assert feat.shape == (B, 320), f"get_feat: {feat.shape}"


# ---------------------------------------------------------------------------
# Test 7: TokenImageDecoder output shape
# ---------------------------------------------------------------------------

def test_image_decoder_shape():
    dec = _image_decoder()
    deter_flat = torch.randn(B, N * D)
    stoch_flat = torch.randn(B, N * Z)
    img = dec(deter_flat, stoch_flat)
    assert img.shape == (B, 3, 64, 64), f"decoder: {img.shape}"


# ---------------------------------------------------------------------------
# Test 8: TokenStateDecoder output shape
# ---------------------------------------------------------------------------

def test_state_decoder_shape():
    dec = _state_decoder()
    feat = torch.randn(B, 320)
    state = dec(feat)
    assert state.shape == (B, 4), f"state decoder: {state.shape}"


# ---------------------------------------------------------------------------
# Test 9: VFE loss compatible with (B, Z) pooled mean/std; backward works
# ---------------------------------------------------------------------------

def test_loss_backward():
    enc = _encoder()
    tr = _transition()
    img_dec = _image_decoder()
    st_dec = _state_decoder()

    s0 = tr.initial(B, device=torch.device("cpu"))
    action = torch.randn(B, 2)
    obs_img = torch.randn(B, 3, 64, 64)
    obs_state = torch.randn(B, 4)

    obs_tokens = enc(obs_img, obs_state)
    post, prior = tr.obs_step(s0, action, obs_tokens)

    feat = tr.get_feat(post)
    recon_img = img_dec(post.deter, post.stoch)
    recon_state = st_dec(feat)

    # compute_vfe receives (B, Z) pooled mean/std — same shape as RSSM.
    loss, info = compute_vfe(
        post.mean, post.std,
        prior.mean, prior.std,
        obs_img, recon_img,
        obs_state, recon_state,
        free_nats=1.0, kl_dyn_scale=1.0, kl_rep_scale=0.5,
    )
    assert loss.isfinite(), f"loss not finite: {loss}"
    loss.backward()

    # Gradients must flow to encoder and transition parameters.
    enc_grad = sum(
        p.grad.abs().sum().item()
        for p in enc.parameters()
        if p.grad is not None
    )
    tr_grad = sum(
        p.grad.abs().sum().item()
        for p in tr.parameters()
        if p.grad is not None
    )
    assert enc_grad > 0, "no gradient in encoder"
    assert tr_grad > 0,  "no gradient in transition"


# ---------------------------------------------------------------------------
# Test 10a: Default config selects RSSM (regression guard)
# ---------------------------------------------------------------------------

def test_default_config_uses_rssm():
    from active_inference.agent import WorldModel
    cfg = Config.from_yaml("configs/default.yaml")
    wm = WorldModel(cfg)
    assert wm._wm_type == "rssm"
    from active_inference.models.rssm import RSSM
    assert isinstance(wm.rssm, RSSM)


# ---------------------------------------------------------------------------
# Test 10b: token_vit config selects TokenViTTransition
# ---------------------------------------------------------------------------

def test_token_vit_config_selects_correct_model():
    from active_inference.agent import WorldModel
    cfg = Config.from_yaml("configs/experiment/token_vit.yaml")
    assert cfg.model.world_model_type == "token_vit"
    wm = WorldModel(cfg)
    assert wm._wm_type == "token_vit"
    assert isinstance(wm.rssm, TokenViTTransition)


# ---------------------------------------------------------------------------
# Test 10c: DeepAIFAgent forward pass with token_vit config
# ---------------------------------------------------------------------------

def test_agent_step_token_vit():
    from active_inference.agent import DeepAIFAgent
    cfg = Config.from_yaml("configs/experiment/token_vit.yaml")
    cfg.device = "cpu"
    agent = DeepAIFAgent(cfg)
    agent.reset()
    img = torch.randn(3, 64, 64)
    state = torch.randn(4)
    action = agent.step(img, state)
    assert action.shape == (2,)
    assert (action >= -1.0).all() and (action <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 10d: DeepAIFAgent.update() backward pass with token_vit config
# ---------------------------------------------------------------------------

def test_agent_update_token_vit():
    from active_inference.agent import DeepAIFAgent
    torch.manual_seed(0)
    cfg = Config.from_yaml("configs/experiment/token_vit.yaml")
    cfg.device = "cpu"
    cfg.training.seq_len = 3
    cfg.training.batch_size = 2
    agent = DeepAIFAgent(cfg)
    T = cfg.training.seq_len
    images = torch.randn(2, T, 3, 64, 64)
    states = torch.randn(2, T, 4)
    actions = torch.randn(2, T, 2).clamp(-1, 1)
    info = agent.update(images, states, actions)
    assert torch.isfinite(torch.tensor(info["total_loss"]))


# ---------------------------------------------------------------------------
# Test 10e: iCEM planner runs without error (expand compatibility)
# ---------------------------------------------------------------------------

def test_icem_expand_compatibility():
    """Confirm iCEM's state.expand(n_samples, -1) works for 2D token state."""
    tr = _transition()
    s0 = tr.initial(1, device=torch.device("cpu"))
    # Simulate what iCEM does: expand initial state to n_samples.
    n_samples = 10
    expanded = RSSMState(*[x.expand(n_samples, -1) for x in s0])
    assert expanded.deter.shape == (n_samples, N * D)
    assert expanded.stoch.shape == (n_samples, N * Z)
    assert expanded.mean.shape  == (n_samples, Z)
    assert expanded.std.shape   == (n_samples, Z)
    # Then imagine runs correctly on expanded state.
    actions = torch.randn(3, n_samples, 2)
    traj = tr.imagine(expanded, actions)
    assert len(traj) == 3
    assert traj[0].deter.shape == (n_samples, N * D)
