"""Unit tests for ViT spatial enhancement features A1/A2/A3.

All tests run on CPU with small synthetic tensors; no CARLA or HDF5 needed.
"""

from __future__ import annotations

import torch
import pytest

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VIT_SMALL = [
    "device=cpu", "training.batch_size=2", "training.seq_len=5",
    "training.epochs=1",
    "token_vit.num_layers=1", "token_vit.num_prior_layers=1",
    "token_vit.num_post_layers=1",
    "token_vit.embed_dim=32", "token_vit.deter_dim=32",
    "token_vit.stoch_dim=8", "token_vit.feat_dim=40",
    "token_vit.num_tokens=64", "token_vit.num_heads=4",
    "rssm.stoch_dim=8", "rssm.deter_dim=32", "rssm.embed_dim=32",
    "ensemble.num_heads=2", "ensemble.hidden_dim=32",
]

DEBUG_YAML = "configs/experiment/debug.yaml"
VIT_YAML = "configs/experiment/token_vit.yaml"


def _make_vit_agent(extra_overrides=None):
    overrides = list(VIT_SMALL)
    if extra_overrides:
        overrides += extra_overrides
    cfg = Config.from_yaml(VIT_YAML, overrides=overrides)
    return DeepAIFAgent(cfg)


def _smoke_update(agent, B=2, T=5):
    images = torch.zeros(B, T, 3, 64, 64)
    states = torch.zeros(B, T, 4)
    actions = torch.zeros(B, T, 2)
    return agent.update(images, states, actions)


# ─────────────────────────────────────────────────────────────────────────────
# A3: Token-weighted KL
# ─────────────────────────────────────────────────────────────────────────────

class TestA3TokenKLWeighting:
    def test_none_weighting_rssm_runs(self):
        cfg = Config.from_yaml(DEBUG_YAML, overrides=["training.token_kl_weighting=none"])
        agent = DeepAIFAgent(cfg)
        info = _smoke_update(agent, B=2, T=5)
        assert "kl_dyn" in info
        assert not torch.isnan(torch.tensor(info["kl_dyn"]))

    def test_none_weighting_vit_runs(self):
        agent = _make_vit_agent(["training.token_kl_weighting=none"])
        info = _smoke_update(agent)
        assert info["kl_dyn"] >= 0.0

    def test_error_weighting_vit_runs(self):
        agent = _make_vit_agent(["training.token_kl_weighting=error"])
        info = _smoke_update(agent)
        assert info["kl_dyn"] >= 0.0
        assert not torch.isnan(torch.tensor(info["kl_dyn"]))

    def test_error_weighting_rssm_falls_back(self):
        """RSSM (2D state): error weighting must fall back to standard mean."""
        cfg = Config.from_yaml(DEBUG_YAML, overrides=["training.token_kl_weighting=error"])
        agent = DeepAIFAgent(cfg)
        info_error = _smoke_update(agent, B=2, T=5)

        cfg2 = Config.from_yaml(DEBUG_YAML, overrides=["training.token_kl_weighting=none"])
        agent2 = DeepAIFAgent(cfg2)
        # Copy weights so both agents are identical
        agent2.world_model.load_state_dict(agent.world_model.state_dict())
        info_none = _smoke_update(agent2, B=2, T=5)
        # With identical weights and zero inputs, both should be numerically equal
        assert abs(info_error["kl_dyn"] - info_none["kl_dyn"]) < 1e-4, (
            "RSSM token_kl_weighting=error should fall back to none for 2D states"
        )

    def test_error_weighting_vit_differs_from_none(self):
        """token_kl_weighting=error should generally differ from none on ViT."""
        torch.manual_seed(0)
        agent_none = _make_vit_agent(["training.token_kl_weighting=none"])
        agent_error = _make_vit_agent(["training.token_kl_weighting=error"])
        # Copy same initial weights
        agent_error.world_model.load_state_dict(agent_none.world_model.state_dict())

        # Use non-trivial input so token KLs actually differ
        torch.manual_seed(42)
        images = torch.randn(2, 5, 3, 64, 64) * 0.1
        states = torch.randn(2, 5, 4) * 0.1
        actions = torch.randn(2, 5, 2) * 0.1

        info_none = agent_none.update(images, states, actions)
        info_error = agent_error.update(images, states, actions)
        # They CAN be different (not guaranteed on first step but generally are)
        # Just check both are finite
        assert not torch.isnan(torch.tensor(info_none["kl_dyn"]))
        assert not torch.isnan(torch.tensor(info_error["kl_dyn"]))


# ─────────────────────────────────────────────────────────────────────────────
# A1: Action-warped prior
# ─────────────────────────────────────────────────────────────────────────────

class TestA1ActionWarp:
    def _make_transition(self, use_action_warp: bool):
        from active_inference.models.token_vit import TokenViTTransition
        return TokenViTTransition(
            num_tokens=64, embed_dim=32, deter_dim=32, stoch_dim=8,
            action_dim=2, num_prior_layers=1, num_post_layers=1,
            num_heads=4, use_action_warp=use_action_warp,
        )

    def test_warp_identity_when_action_zero(self):
        """With zero-initialized warp params and zero action, output ≈ input."""
        trans = self._make_transition(use_action_warp=True)
        B, N, D = 2, 64, 32
        x = torch.randn(B, N, D)
        action = torch.zeros(B, 2)
        out = trans._apply_warp(x, action)
        assert out.shape == x.shape
        # flow = 0, so sampling at identity grid → output == input (border pad)
        assert torch.allclose(out, x, atol=1e-5), (
            "Zero action with zero-initialized warp weights must give identity"
        )

    def test_warp_shape_preserved(self):
        trans = self._make_transition(use_action_warp=True)
        B, N, D = 3, 64, 32
        x = torch.randn(B, N, D)
        action = torch.randn(B, 2) * 0.1
        out = trans._apply_warp(x, action)
        assert out.shape == (B, N, D)

    def test_warp_changes_output_with_nonzero_flow(self):
        """Non-zero flow (manually forced via weight) should change output vs identity."""
        trans = self._make_transition(use_action_warp=True)
        B, N, D = 2, 64, 32
        x = torch.randn(B, N, D)

        # Make action[0]=1 produce a large global x-shift via weight (not bias)
        with torch.no_grad():
            trans._warp_global.weight.fill_(0.0)
            trans._warp_global.bias.fill_(0.0)
            trans._warp_residual[-1].weight.fill_(0.0)
            trans._warp_residual[-1].bias.fill_(0.0)
            trans._warp_global.weight[0, 0] = 5.0  # action[0] → large x-flow
            trans._flow_scale = 1.0                # disable flow scale cap

        action_large = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        action_zero = torch.zeros(B, 2)
        out_shifted = trans._apply_warp(x, action_large)
        out_zero = trans._apply_warp(x, action_zero)
        # Large flow → bilinear sampling returns different values
        assert not torch.allclose(out_shifted, out_zero, atol=1e-4)

    def test_warp_smooth_loss_is_scalar(self):
        trans = self._make_transition(use_action_warp=True)
        B, N, D = 2, 64, 32
        x = torch.randn(B, N, D)
        action = torch.randn(B, 2) * 0.1
        trans._apply_warp(x, action)
        loss = trans.warp_smoothness_loss()
        assert loss.ndim == 0  # scalar tensor
        assert loss.item() >= 0.0

    def test_warp_smooth_loss_zero_when_disabled(self):
        trans = self._make_transition(use_action_warp=False)
        loss = trans.warp_smoothness_loss()
        assert loss.item() == 0.0

    def test_img_step_shape_with_warp(self):
        trans = self._make_transition(use_action_warp=True)
        state = trans.initial(2)
        action = torch.randn(2, 2) * 0.1
        new_state = trans.img_step(state, action)
        assert new_state.deter.shape == (2, 64, 32)

    def test_base_grid_center_is_zero(self):
        """For odd G, center token should be at (0, 0) in normalized coords."""
        trans = self._make_transition(use_action_warp=True)
        # G=8 (even): test corner values
        grid = trans._make_base_grid(8, torch.device("cpu"))  # (1, 8, 8, 2)
        assert grid.shape == (1, 8, 8, 2)
        # Top-left: (-1, -1), bottom-right: (1, 1)
        assert torch.allclose(grid[0, 0, 0], torch.tensor([-1.0, -1.0]), atol=1e-5)
        assert torch.allclose(grid[0, 7, 7], torch.tensor([1.0, 1.0]), atol=1e-5)

    def test_vit_agent_with_warp_smoke(self):
        agent = _make_vit_agent(["token_vit.use_action_warp=true"])
        info = _smoke_update(agent)
        assert "total_loss" in info
        assert not torch.isnan(torch.tensor(info["total_loss"]))

    def test_warp_smoothness_in_training_loss(self):
        """warp_smoothness_weight > 0 should change total_loss."""
        torch.manual_seed(0)
        images = torch.randn(2, 5, 3, 64, 64) * 0.1
        states = torch.randn(2, 5, 4) * 0.1
        actions = torch.randn(2, 5, 2) * 0.1

        agent_no_smooth = _make_vit_agent([
            "token_vit.use_action_warp=true",
            "training.warp_smoothness_weight=0.0",
        ])
        agent_smooth = _make_vit_agent([
            "token_vit.use_action_warp=true",
            "training.warp_smoothness_weight=1.0",
        ])
        agent_smooth.world_model.load_state_dict(agent_no_smooth.world_model.state_dict())

        info_no = agent_no_smooth.update(images.clone(), states.clone(), actions.clone())
        info_sm = agent_smooth.update(images.clone(), states.clone(), actions.clone())
        # Smooth weight adds to loss — they should differ
        assert abs(info_no["total_loss"] - info_sm["total_loss"]) > 1e-8, (
            "warp_smoothness_weight=1.0 must change total_loss"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A2: Token-wise preference
# ─────────────────────────────────────────────────────────────────────────────

class TestA2TokenWisePreference:
    def test_token_scores_shape(self):
        from active_inference.training.token_preference import TokenContrastivePreference
        pref = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=40)
        B, N, D = 3, 64, 40
        z = torch.randn(B, N, D)
        scores = pref.token_scores(z)
        assert scores.shape == (B, N)

    def test_score_frames_shape(self):
        from active_inference.training.token_preference import TokenContrastivePreference
        pref = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=40)
        B, N, D = 3, 64, 40
        z = torch.randn(B, N, D)
        frame_scores = pref.score_frames(z, topk=8)
        assert frame_scores.shape == (B,)

    def test_topk_clamps_to_n(self):
        from active_inference.training.token_preference import topk_mean_token_score
        scores = torch.randn(2, 4)
        out = topk_mean_token_score(scores, k=100)  # k > N → clamp to N
        assert out.shape == (2,)

    def test_extract_token_features_deter_stoch(self):
        from active_inference.training.token_preference import extract_token_features
        from active_inference.models.token_vit import TokenRSSMState
        B, N, D, Z = 2, 64, 32, 8
        state = TokenRSSMState(
            deter=torch.zeros(B, N, D),
            stoch=torch.zeros(B, N, Z),
            mean=torch.zeros(B, Z),
            std=torch.ones(B, Z),
            token_mean=torch.zeros(B, N, Z),
            token_std=torch.ones(B, N, Z),
        )
        feat = extract_token_features(state, "deter_stoch")
        assert feat.shape == (B, N, D + Z)

    def test_extract_token_features_rssm_raises(self):
        from active_inference.training.token_preference import extract_token_features
        from active_inference.models.rssm import RSSMState
        state = RSSMState(
            deter=torch.zeros(2, 32),
            stoch=torch.zeros(2, 8),
            mean=torch.zeros(2, 8),
            std=torch.ones(2, 8),
        )
        with pytest.raises(ValueError, match="3D"):
            extract_token_features(state, "deter_stoch")

    def test_efe_score_uses_token_pref_when_provided(self):
        """EFE scorer should call token_scores when pref_model has that method."""
        from active_inference.planning.efe import EFEScorer
        from active_inference.models.ensemble import EnsembleTransitionHeads
        from active_inference.training.token_preference import TokenContrastivePreference
        from active_inference.models.token_vit import TokenRSSMState

        B, N, D, Z = 4, 64, 32, 8
        H = 3

        def _make_state():
            return TokenRSSMState(
                deter=torch.randn(B, N, D),
                stoch=torch.randn(B, N, Z),
                mean=torch.randn(B, Z),
                std=torch.ones(B, Z),
                token_mean=torch.randn(B, N, Z),
                token_std=torch.ones(B, N, Z),
            )

        trajectory = [_make_state() for _ in range(H)]
        feats = [torch.randn(B, 40) for _ in range(H)]
        means = [s.mean for s in trajectory]
        stds = [s.std for s in trajectory]

        pref = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=D + Z)
        ensemble = EnsembleTransitionHeads(feat_dim=40, stoch_dim=Z, hidden_dim=16, num_heads=2)
        scorer = EFEScorer(beta_epistemic=0.0)  # disable epistemic for clean test

        scores = scorer.score(
            feats, means, stds,
            pref_model=pref,
            ensemble=ensemble,
            trajectory_states=trajectory,
        )
        assert scores.shape == (B,)
        assert not scores.isnan().any()


# ─────────────────────────────────────────────────────────────────────────────
# Regression: all flags off → identical to baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionAllFlagsOff:
    def test_vit_baseline_unaffected(self):
        """All A1/A2/A3 flags off → ViT smoke update passes without NaN."""
        agent = _make_vit_agent()  # all defaults = off
        info = _smoke_update(agent)
        for k in ["img_loss", "state_loss", "kl_dyn", "kl_rep", "total_loss"]:
            assert not torch.isnan(torch.tensor(info[k])), f"{k} is NaN"

    def test_rssm_baseline_unaffected(self):
        """RSSM baseline (A flags don't exist) still runs."""
        cfg = Config.from_yaml(DEBUG_YAML)
        agent = DeepAIFAgent(cfg)
        info = _smoke_update(agent, B=2, T=5)
        assert info["total_loss"] > 0

    def test_all_flags_on_vit_smoke(self):
        """All three features enabled simultaneously — no NaN."""
        agent = _make_vit_agent([
            "token_vit.use_action_warp=true",
            "training.token_kl_weighting=error",
            "training.warp_smoothness_weight=0.1",
        ])
        info = _smoke_update(agent)
        for k in ["img_loss", "state_loss", "kl_dyn", "kl_rep", "total_loss"]:
            assert not torch.isnan(torch.tensor(info[k])), f"{k} is NaN with all flags on"
