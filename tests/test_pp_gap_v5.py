"""Unit tests for pp_gap_v5.py.

Tests cover:
  - imagine_from_states: horizon-h prior state is h img_steps ahead
  - compute_pp_gap: Δ_post, Δ_prior, PP_Gap, Retention match manual values
  - encode_posterior: episode boundary reset behaviour
  - ContrastiveScorer: log_prob ratio computation
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pp_gap_v5 import (
    ContrastiveScorer,
    _gmm_log_prob,
    compute_pp_gap,
    encode_posterior,
    imagine_from_states,
)


# Patch Normal.rsample → return mean so img_step / obs_step are deterministic.
# This lets us compare two separate rollouts for exact equality.
def _det_rsample(self, sample_shape=torch.Size()):
    return self.mean.expand(sample_shape + self.mean.shape) if sample_shape else self.mean


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rssm_agent(stoch_dim: int = 8, deter_dim: int = 32):
    from active_inference.config import Config
    from active_inference.agent import WorldModel
    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    wm = WorldModel(cfg)
    wm.eval()
    return wm, cfg


def _make_state(wm, batch: int = 1):
    """Return initial RSSM state."""
    return wm.rssm.initial(batch, torch.device("cpu"))


def _make_scalar_scorer(clean_offset: float = 1.0, avoid_offset: float = -1.0):
    """Build a mock ContrastiveScorer that returns a known constant score."""
    class _MockScorer:
        def __call__(self, state):
            # Returns a fixed scalar independent of state
            z = state.mean
            return torch.full((z.shape[0],), 0.5, dtype=torch.float32)
    return _MockScorer()


def _make_real_scorer(stoch_dim: int = 8):
    """Build a ContrastiveScorer from synthetic GMM parameters."""
    K_clean, K_avoid = 2, 2
    ckpt = {
        "contrastive_preference": {
            "clean_means":    torch.randn(K_clean, stoch_dim),
            "clean_log_stds": torch.zeros(K_clean, stoch_dim),
            "clean_logits":   torch.zeros(K_clean),
            "avoid_means":    torch.randn(K_avoid, stoch_dim),
            "avoid_log_stds": torch.zeros(K_avoid, stoch_dim),
            "avoid_logits":   torch.zeros(K_avoid),
            "contrast_scale": 1.0,
        }
    }
    return ContrastiveScorer(ckpt, device=torch.device("cpu"))


def _synthetic_data(T: int = 20, n_eps: int = 2, stoch_dim: int = 8):
    """Synthetic image/state/action/episode_ids tensors (T must be divisible by n_eps)."""
    assert T % n_eps == 0, "T must be divisible by n_eps for clean episode splitting"
    images = torch.zeros(T, 3, 64, 64)
    states = torch.zeros(T, 4)
    actions = torch.zeros(T, 2)
    ep_ids = np.repeat(np.arange(n_eps), T // n_eps)
    return images, states, actions, ep_ids


# ─────────────────────────────────────────────────────────────────────────────
# _gmm_log_prob
# ─────────────────────────────────────────────────────────────────────────────

class TestGmmLogProb:

    def test_output_shape(self):
        z = torch.randn(10, 8)
        means = torch.randn(3, 8)
        log_stds = torch.zeros(3, 8)
        logits = torch.zeros(3)
        lp = _gmm_log_prob(z, means, log_stds, logits)
        assert lp.shape == (10,)

    def test_higher_at_component_mean(self):
        """Single-component GMM: log_prob is maximized at the mean."""
        mean = torch.zeros(1, 8)
        log_std = torch.zeros(1, 8) - 1.0  # tight std
        logits = torch.zeros(1)
        at_mean = _gmm_log_prob(mean, mean, log_std, logits)
        at_far = _gmm_log_prob(mean + 5.0, mean, log_std, logits)
        assert at_mean.item() > at_far.item()

    def test_batch_3d_input(self):
        """Accepts (S, B, D) input."""
        z = torch.randn(4, 10, 8)
        means = torch.randn(3, 8)
        lp = _gmm_log_prob(z, means, torch.zeros(3, 8), torch.zeros(3))
        assert lp.shape == (4, 10)


# ─────────────────────────────────────────────────────────────────────────────
# ContrastiveScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastiveScorer:

    def test_score_shape(self):
        scorer = _make_real_scorer(stoch_dim=8)
        from active_inference.config import Config
        from active_inference.agent import WorldModel
        cfg = Config.from_yaml("configs/experiment/debug.yaml")
        wm = WorldModel(cfg)
        state = wm.rssm.initial(5, torch.device("cpu"))
        scores = scorer(state)
        assert scores.shape == (5,)

    def test_score_is_finite(self):
        scorer = _make_real_scorer()
        from active_inference.config import Config
        from active_inference.agent import WorldModel
        cfg = Config.from_yaml("configs/experiment/debug.yaml")
        wm = WorldModel(cfg)
        state = wm.rssm.initial(3, torch.device("cpu"))
        assert scorer(state).isfinite().all()

    def test_clean_state_scores_higher_than_avoid(self):
        """Clean-biased state should score higher than avoid-biased state."""
        stoch_dim = 8
        K = 2
        clean_means = torch.zeros(K, stoch_dim) + 3.0
        avoid_means = torch.zeros(K, stoch_dim) - 3.0
        ckpt = {
            "contrastive_preference": {
                "clean_means":    clean_means,
                "clean_log_stds": torch.zeros(K, stoch_dim) - 1.0,
                "clean_logits":   torch.zeros(K),
                "avoid_means":    avoid_means,
                "avoid_log_stds": torch.zeros(K, stoch_dim) - 1.0,
                "avoid_logits":   torch.zeros(K),
                "contrast_scale": 1.0,
            }
        }
        scorer = ContrastiveScorer(ckpt, torch.device("cpu"))
        from active_inference.models.rssm import RSSMState

        z_clean = torch.zeros(1, stoch_dim) + 3.0   # near clean centre
        z_avoid = torch.zeros(1, stoch_dim) - 3.0   # near avoid centre

        h_clean = torch.zeros(1, 32)
        state_clean = RSSMState(deter=h_clean, stoch=z_clean, mean=z_clean,
                                std=torch.ones_like(z_clean))
        state_avoid = RSSMState(deter=h_clean, stoch=z_avoid, mean=z_avoid,
                                std=torch.ones_like(z_avoid))

        score_c = scorer(state_clean).item()
        score_a = scorer(state_avoid).item()
        assert score_c > score_a, (
            f"Clean state should score higher: {score_c:.3f} vs {score_a:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# imagine_from_states
# ─────────────────────────────────────────────────────────────────────────────

class TestImagineFromStates:

    def test_horizon_h_equals_h_img_steps(self):
        """Imagined state after h steps equals h calls to img_step (deterministic path)."""
        wm, _ = _make_rssm_agent()
        init = _make_state(wm, batch=1)
        h = 3

        with patch.object(Normal, 'rsample', _det_rsample):
            # Manual h-step rollout
            state = init
            with torch.no_grad():
                for _ in range(h):
                    state = wm.rssm.img_step(state, torch.zeros(1, 2))
            expected_deter = state.deter.detach()

            # imagine_from_states rollout
            with torch.no_grad():
                imagined = imagine_from_states(
                    wm, [init], horizon=h, device=torch.device("cpu"),
                    action_mode="zero",
                )

        assert len(imagined) == 1
        assert torch.allclose(imagined[0].deter, expected_deter, atol=1e-5), (
            "Imagined deter after h steps must match h manual img_step calls"
        )

    def test_output_count_equals_input_count(self):
        wm, _ = _make_rssm_agent()
        n_states = 7
        initial_states = [_make_state(wm, batch=1) for _ in range(n_states)]
        with torch.no_grad():
            out = imagine_from_states(wm, initial_states, horizon=5,
                                      device=torch.device("cpu"))
        assert len(out) == n_states

    def test_horizon_zero_returns_initial_state(self):
        """h=0 should leave the state unchanged."""
        wm, _ = _make_rssm_agent()
        init = _make_state(wm, batch=1)
        with torch.no_grad():
            out = imagine_from_states(wm, [init], horizon=0,
                                      device=torch.device("cpu"))
        assert torch.allclose(out[0].deter, init.deter, atol=1e-5)

    def test_different_horizons_give_different_states(self):
        wm, _ = _make_rssm_agent()
        init = _make_state(wm, batch=1)
        with torch.no_grad():
            out3 = imagine_from_states(wm, [init], horizon=3,
                                       device=torch.device("cpu"))
            out7 = imagine_from_states(wm, [init], horizon=7,
                                       device=torch.device("cpu"))
        assert not torch.allclose(out3[0].deter, out7[0].deter, atol=1e-3), (
            "Different horizons should give different states"
        )


# ─────────────────────────────────────────────────────────────────────────────
# encode_posterior
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodePosterior:

    def test_output_count_matches_input_frames(self):
        wm, _ = _make_rssm_agent()
        T = 16  # divisible by n_eps=2
        images, states, actions, ep_ids = _synthetic_data(T, n_eps=2)
        with torch.no_grad():
            posts = encode_posterior(wm, images, states, actions, ep_ids,
                                     torch.device("cpu"), max_samples=T)
        assert len(posts) == T

    def test_max_samples_limits_output(self):
        wm, _ = _make_rssm_agent()
        T = 20
        images, states, actions, ep_ids = _synthetic_data(T, n_eps=1)
        with torch.no_grad():
            posts = encode_posterior(wm, images, states, actions, ep_ids,
                                     torch.device("cpu"), max_samples=10)
        assert len(posts) == 10

    def test_episode_boundary_resets_deter(self):
        """Deter immediately after a boundary must match fresh initial state."""
        wm, _ = _make_rssm_agent()
        T = 10
        images = torch.zeros(T, 3, 64, 64)
        states = torch.zeros(T, 4)
        actions = torch.zeros(T, 2)
        # Episode boundary at frame 5
        ep_ids = np.array([0]*5 + [1]*5)

        with torch.no_grad():
            posts = encode_posterior(wm, images, states, actions, ep_ids,
                                     torch.device("cpu"), max_samples=T)

        # The state at frame 5 should be computed from a fresh initial state
        fresh_init = wm.rssm.initial(1, torch.device("cpu"))
        fresh_embed = wm.encode_obs(images[5:6], states[5:6])
        fresh_post, _ = wm.rssm.obs_step(
            fresh_init, torch.zeros(1, 2), fresh_embed,
        )
        assert torch.allclose(posts[5].mean, fresh_post.mean.detach(), atol=1e-5), (
            "State at episode boundary must match fresh initial-state rollout"
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_pp_gap  — manual calculation check
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePPGap:

    def _setup(self, T: int = 20):
        wm, cfg = _make_rssm_agent()
        scorer = _make_scalar_scorer()  # always returns 0.5
        images, states, actions, ep_ids = _synthetic_data(T, n_eps=2)
        return wm, scorer, images, states, actions, ep_ids

    def test_delta_post_zero_when_scorer_constant(self):
        """If scorer always returns 0.5, Δ_post must be 0."""
        wm, scorer, images, states, actions, ep_ids = self._setup()
        with torch.no_grad():
            res = compute_pp_gap(
                wm=wm,
                scorer=scorer,
                clean_images=images,
                clean_states=states,
                clean_actions=actions,
                clean_episode_ids=ep_ids,
                obs_images=images,
                obs_states=states,
                obs_actions=actions,
                obs_episode_ids=ep_ids,
                horizons=[1, 3],
                device=torch.device("cpu"),
                max_samples=20,
            )
        assert abs(res["delta_post"]) < 1e-5

    def test_pp_gap_zero_when_delta_post_zero(self):
        """PP_Gap = Δ_post - Δ_prior; both zero → PP_Gap = 0."""
        wm, scorer, images, states, actions, ep_ids = self._setup()
        with torch.no_grad():
            res = compute_pp_gap(
                wm=wm, scorer=scorer,
                clean_images=images, clean_states=states, clean_actions=actions,
                clean_episode_ids=ep_ids,
                obs_images=images, obs_states=states, obs_actions=actions,
                obs_episode_ids=ep_ids,
                horizons=[1, 2, 5], device=torch.device("cpu"), max_samples=20,
            )
        for gap in res["pp_gap"]:
            assert abs(gap) < 1e-5, f"Expected PP_Gap≈0, got {gap}"

    def test_output_keys_present(self):
        wm, scorer, images, states, actions, ep_ids = self._setup()
        with torch.no_grad():
            res = compute_pp_gap(
                wm=wm, scorer=scorer,
                clean_images=images, clean_states=states, clean_actions=actions,
                clean_episode_ids=ep_ids,
                obs_images=images, obs_states=states, obs_actions=actions,
                obs_episode_ids=ep_ids,
                horizons=[1, 3, 5], device=torch.device("cpu"), max_samples=10,
            )
        for key in ("delta_post", "horizons", "delta_prior", "pp_gap", "retention"):
            assert key in res, f"Missing key: {key}"

    def test_retention_nan_when_delta_post_zero(self):
        """Retention = Δ_prior / Δ_post; when Δ_post≈0, retention should be NaN."""
        wm, scorer, images, states, actions, ep_ids = self._setup()
        with torch.no_grad():
            res = compute_pp_gap(
                wm=wm, scorer=scorer,
                clean_images=images, clean_states=states, clean_actions=actions,
                clean_episode_ids=ep_ids,
                obs_images=images, obs_states=states, obs_actions=actions,
                obs_episode_ids=ep_ids,
                horizons=[1], device=torch.device("cpu"), max_samples=10,
            )
        # delta_post ≈ 0 → retention should be nan
        assert math.isnan(res["retention"][0])

    def test_manual_delta_post_calculation(self):
        """Δ_post = mean(clean_post_scores) - mean(obs_post_scores) matches manual.

        Uses deterministic rsample (returns mean) so both compute_pp_gap and the
        manual encode_posterior calls produce identical latent sequences.
        """
        wm, cfg = _make_rssm_agent()
        scorer = _make_real_scorer(stoch_dim=cfg.rssm.stoch_dim)
        T = 10
        clean_images = torch.zeros(T, 3, 64, 64)
        clean_states = torch.zeros(T, 4)
        clean_actions = torch.zeros(T, 2)
        clean_ep_ids = np.zeros(T, dtype=np.int64)

        obs_images = torch.ones(T, 3, 64, 64) * 0.5
        obs_states = torch.ones(T, 4) * 0.1
        obs_actions = torch.zeros(T, 2)
        obs_ep_ids = np.zeros(T, dtype=np.int64)

        with patch.object(Normal, 'rsample', _det_rsample), torch.no_grad():
            res = compute_pp_gap(
                wm=wm, scorer=scorer,
                clean_images=clean_images, clean_states=clean_states,
                clean_actions=clean_actions, clean_episode_ids=clean_ep_ids,
                obs_images=obs_images, obs_states=obs_states,
                obs_actions=obs_actions, obs_episode_ids=obs_ep_ids,
                horizons=[1], device=torch.device("cpu"), max_samples=T,
            )

            # Manual computation of delta_post under the same deterministic patch
            clean_posts = encode_posterior(
                wm, clean_images, clean_states, clean_actions, clean_ep_ids,
                torch.device("cpu"), max_samples=T,
            )
            obs_posts = encode_posterior(
                wm, obs_images, obs_states, obs_actions, obs_ep_ids,
                torch.device("cpu"), max_samples=T,
            )
            clean_s = torch.stack([scorer(s).squeeze() for s in clean_posts])
            obs_s   = torch.stack([scorer(s).squeeze() for s in obs_posts])
            manual_delta = float(clean_s.mean().item() - obs_s.mean().item())

        assert abs(res["delta_post"] - manual_delta) < 1e-5, (
            f"delta_post mismatch: {res['delta_post']:.6f} vs manual {manual_delta:.6f}"
        )

    def test_horizons_list_matches_output(self):
        wm, scorer, images, states, actions, ep_ids = self._setup()
        horizons = [1, 3, 5, 7]
        with torch.no_grad():
            res = compute_pp_gap(
                wm=wm, scorer=scorer,
                clean_images=images, clean_states=states, clean_actions=actions,
                clean_episode_ids=ep_ids,
                obs_images=images, obs_states=states, obs_actions=actions,
                obs_episode_ids=ep_ids,
                horizons=horizons, device=torch.device("cpu"), max_samples=20,
            )
        assert res["horizons"] == horizons
        assert len(res["delta_prior"]) == len(horizons)
        assert len(res["pp_gap"]) == len(horizons)
        assert len(res["retention"]) == len(horizons)
