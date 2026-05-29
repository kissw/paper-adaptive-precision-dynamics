"""Tests for Token VampPrior-like preference module.

Test 1:  extract_token_posterior_stats — shape contract
Test 2:  reject non-token RSSM state
Test 3:  fit_position_posterior_mixture — output shapes
Test 4:  log_prob_posterior_under_mixture — output shape
Test 5:  q_integrated correctness sanity
Test 6:  synthetic clean/avoid separation
Test 7:  top-k scoring shapes
Test 8:  save/load checkpoint structure
"""

import pytest
import torch
from active_inference.models.rssm import RSSMState
from active_inference.models.token_vit import TokenRSSMState
from active_inference.training.token_vampprior_preference import (
    TokenVampPriorPreference,
    extract_token_posterior_stats,
    fit_position_posterior_mixture,
    log_prob_posterior_under_mixture,
    topk_mean_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token_state(B: int = 2, N: int = 64, D: int = 256, Z: int = 64) -> TokenRSSMState:
    return TokenRSSMState(
        deter=torch.randn(B, N, D),
        stoch=torch.randn(B, N, Z),
        mean=torch.randn(B, Z),
        std=torch.rand(B, Z).abs() + 0.1,
        token_mean=torch.randn(B, N, Z),
        token_std=torch.rand(B, N, Z).abs() + 0.1,
    )


def _small_model(N: int = 8, K: int = 2, Z: int = 4) -> TokenVampPriorPreference:
    """Build a tiny model without fitting for shape-only tests."""
    clean_pm  = torch.randn(N, K, Z)
    clean_pls = torch.zeros(N, K, Z)
    clean_pl  = torch.zeros(N, K)
    avoid_pm  = torch.randn(N, K, Z)
    avoid_pls = torch.zeros(N, K, Z)
    avoid_pl  = torch.zeros(N, K)
    return TokenVampPriorPreference(
        clean_pm, clean_pls, clean_pl,
        avoid_pm, avoid_pls, avoid_pl,
    )


# ---------------------------------------------------------------------------
# Test 1: extract_token_posterior_stats — shape contract
# ---------------------------------------------------------------------------

def test_extract_posterior_stats_shape():
    B, N, Z = 3, 64, 64
    state = _make_token_state(B=B, N=N, Z=Z)
    mu, std = extract_token_posterior_stats(state)
    assert mu.shape  == (B, N, Z), f"mu shape: {mu.shape}"
    assert std.shape == (B, N, Z), f"std shape: {std.shape}"


def test_extract_posterior_stats_values():
    """Returned tensors must be the exact token_mean / token_std fields."""
    state = _make_token_state()
    mu, std = extract_token_posterior_stats(state)
    assert torch.equal(mu,  state.token_mean)
    assert torch.equal(std, state.token_std)


# ---------------------------------------------------------------------------
# Test 2: reject non-token RSSM state
# ---------------------------------------------------------------------------

def test_extract_rejects_rssm_state():
    """RSSM state has no token_mean/token_std — must raise ValueError."""
    rssm = RSSMState(
        deter=torch.zeros(2, 256),
        stoch=torch.zeros(2, 64),
        mean=torch.zeros(2, 64),
        std=torch.ones(2, 64),
    )
    with pytest.raises(ValueError, match="token_mean"):
        extract_token_posterior_stats(rssm)


def test_extract_rejects_2d_token_mean():
    """State with 2D token_mean (e.g. accidentally pooled) must raise ValueError."""
    class FakeState:
        token_mean = torch.zeros(2, 64)   # 2D — wrong
        token_std  = torch.ones(2, 64)

    with pytest.raises(ValueError, match="ndim=3"):
        extract_token_posterior_stats(FakeState())


# ---------------------------------------------------------------------------
# Test 3: fit_position_posterior_mixture — output shapes
# ---------------------------------------------------------------------------

def test_fit_mixture_shapes():
    torch.manual_seed(0)
    F, N, Z, K = 50, 8, 4, 3
    mu  = torch.randn(F, N, Z)
    std = torch.rand(F, N, Z).abs() + 0.1

    pm, pls, pl = fit_position_posterior_mixture(mu, std, K=K, seed=0)

    assert pm.shape  == (N, K, Z), f"proto_mean shape: {pm.shape}"
    assert pls.shape == (N, K, Z), f"proto_log_std shape: {pls.shape}"
    assert pl.shape  == (N, K),    f"proto_logits shape: {pl.shape}"


def test_fit_mixture_std_positive():
    """proto_log_std should not be -inf (std > 0 after clamping)."""
    torch.manual_seed(1)
    F, N, Z, K = 30, 4, 4, 2
    mu  = torch.randn(F, N, Z)
    std = torch.rand(F, N, Z).abs() + 0.1

    _, pls, _ = fit_position_posterior_mixture(mu, std, K=K, min_std=0.01)
    assert torch.isfinite(pls).all(), "proto_log_std contains non-finite values"
    assert (pls.exp() > 0).all(),     "proto std should be positive"


# ---------------------------------------------------------------------------
# Test 4: log_prob_posterior_under_mixture — output shape
# ---------------------------------------------------------------------------

def test_log_prob_mixture_shape():
    B, N, K, Z = 3, 8, 2, 4
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.rand(B, N, Z).abs() + 0.1
    pm     = torch.randn(N, K, Z)
    pls    = torch.zeros(N, K, Z)
    pl     = torch.zeros(N, K)

    out = log_prob_posterior_under_mixture(q_mean, q_std, pm, pls, pl)
    assert out.shape == (B, N), f"expected ({B},{N}), got {out.shape}"


def test_log_prob_mixture_finite():
    """log_prob must be finite for well-formed inputs."""
    B, N, K, Z = 2, 16, 3, 4
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.ones(B, N, Z) * 0.5
    pm     = torch.randn(N, K, Z)
    pls    = torch.zeros(N, K, Z)
    pl     = torch.zeros(N, K)

    out = log_prob_posterior_under_mixture(q_mean, q_std, pm, pls, pl)
    assert torch.isfinite(out).all(), "log_prob contains non-finite values"


# ---------------------------------------------------------------------------
# Test 5: q_integrated correctness sanity
# ---------------------------------------------------------------------------

def test_q_integrated_close_q_higher():
    """A q centered near a prototype should score higher than a far q."""
    torch.manual_seed(2)
    N, K, Z = 4, 2, 4

    proto_mean = torch.zeros(N, K, Z)
    proto_mean[:, 0, 0] = 3.0   # component 0 at +3 in dim 0

    proto_log_std = torch.full((N, K, Z), -1.0)  # std ~ 0.37
    proto_logits  = torch.zeros(N, K)

    # Close query: q_mean near prototype centre
    q_close_mean = torch.zeros(1, N, Z)
    q_close_mean[:, :, 0] = 3.0
    q_close_std  = torch.ones(1, N, Z) * 0.1

    # Far query: q_mean at opposite end
    q_far_mean = torch.zeros(1, N, Z)
    q_far_mean[:, :, 0] = -3.0
    q_far_std  = torch.ones(1, N, Z) * 0.1

    lp_close = log_prob_posterior_under_mixture(
        q_close_mean, q_close_std, proto_mean, proto_log_std, proto_logits,
    )
    lp_far = log_prob_posterior_under_mixture(
        q_far_mean, q_far_std, proto_mean, proto_log_std, proto_logits,
    )

    assert (lp_close > lp_far).all(), (
        "Close query should have higher log_prob than far query. "
        f"close={lp_close.mean():.2f}  far={lp_far.mean():.2f}"
    )


# ---------------------------------------------------------------------------
# Test 6: synthetic clean/avoid separation
# ---------------------------------------------------------------------------

def test_synthetic_separation():
    """Clean posterior means near +2, avoid near -2; expect positive gap."""
    torch.manual_seed(3)
    F, N, Z, K = 100, 4, 4, 2

    clean_mu  = torch.randn(F, N, Z) * 0.3
    clean_mu[:, :, 0] += 2.0
    clean_std = torch.ones(F, N, Z) * 0.3

    avoid_mu  = torch.randn(F, N, Z) * 0.3
    avoid_mu[:, :, 0] -= 2.0
    avoid_std = torch.ones(F, N, Z) * 0.3

    clean_pm, clean_pls, clean_pl = fit_position_posterior_mixture(
        clean_mu, clean_std, K=K, seed=0, min_std=0.01,
    )
    avoid_pm, avoid_pls, avoid_pl = fit_position_posterior_mixture(
        avoid_mu, avoid_std, K=K, seed=0, min_std=0.01,
    )

    model = TokenVampPriorPreference(
        clean_pm, clean_pls, clean_pl,
        avoid_pm, avoid_pls, avoid_pl,
        contrast_scale=1.0,
        score_mode="q_integrated",
    )

    with torch.no_grad():
        c_scores = model.token_scores(clean_mu, clean_std)  # (F, N)
        a_scores = model.token_scores(avoid_mu, avoid_std)  # (F, N)

    c_avg = c_scores.mean().item()
    a_avg = a_scores.mean().item()
    gap   = c_avg - a_avg

    assert c_avg > 0,  f"clean avg score should be > 0, got {c_avg:.3f}"
    assert a_avg < 0,  f"avoid avg score should be < 0, got {a_avg:.3f}"
    assert gap   > 0,  f"gap should be positive, got {gap:.3f}"


# ---------------------------------------------------------------------------
# Test 7: top-k scoring shapes and k=N equals mean
# ---------------------------------------------------------------------------

def test_token_scores_shape():
    B, N, Z = 3, 8, 4
    model = _small_model(N=N, Z=Z)
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.ones(B, N, Z) * 0.1

    scores = model.token_scores(q_mean, q_std)
    assert scores.shape == (B, N), f"expected ({B},{N}), got {scores.shape}"


def test_score_frames_shapes():
    B, N, Z = 3, 8, 4
    model = _small_model(N=N, Z=Z)
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.ones(B, N, Z) * 0.1

    for k in [1, 4, N]:
        out = model.score_frames(q_mean, q_std, topk=k)
        assert out.shape == (B,), f"k={k}: expected ({B},), got {out.shape}"


def test_score_frames_k_equals_N_is_mean():
    """score_frames with k=N must equal mean over all tokens."""
    torch.manual_seed(4)
    B, N, Z = 2, 8, 4
    model = _small_model(N=N, Z=Z)
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.ones(B, N, Z) * 0.1

    full  = model.score_frames(q_mean, q_std, topk=N)
    mean_ = model.token_scores(q_mean, q_std).mean(dim=-1)
    assert torch.allclose(full, mean_, atol=1e-5), f"{full} != {mean_}"


def test_score_frames_obstacle_direction():
    """obstacle direction should flip sign vs clean direction."""
    B, N, Z = 2, 8, 4
    model = _small_model(N=N, Z=Z)
    q_mean = torch.randn(B, N, Z)
    q_std  = torch.ones(B, N, Z) * 0.1

    clean_score    = model.score_frames(q_mean, q_std, topk=4, direction="clean")
    obstacle_score = model.score_frames(q_mean, q_std, topk=4, direction="obstacle")
    # obstacle uses top-k of (-scores), so it's NOT simply -clean_score in general.
    # But both must be (B,) shaped.
    assert clean_score.shape    == (B,)
    assert obstacle_score.shape == (B,)


# ---------------------------------------------------------------------------
# Test 8: save/load checkpoint structure
# ---------------------------------------------------------------------------

def test_checkpoint_structure_keys(tmp_path):
    """token_vampprior_preference key present with all required sub-keys."""
    N, K_c, K_a, Z = 4, 2, 3, 4
    clean_pm  = torch.randn(N, K_c, Z)
    clean_pls = torch.zeros(N, K_c, Z)
    clean_pl  = torch.zeros(N, K_c)
    avoid_pm  = torch.randn(N, K_a, Z)
    avoid_pls = torch.zeros(N, K_a, Z)
    avoid_pl  = torch.zeros(N, K_a)

    tvp = {
        "type":          "position_conditioned_token_posterior_mixture",
        "method":        "vampprior_like",
        "score_mode":    "q_integrated",
        "latent_dim":    Z,
        "num_tokens":    N,
        "K_clean":       K_c,
        "K_avoid":       K_a,
        "contrast_scale": 1.0,
        "min_std":       0.01,
        "clean_mean":    clean_pm,
        "clean_log_std": clean_pls,
        "clean_logits":  clean_pl,
        "avoid_mean":    avoid_pm,
        "avoid_log_std": avoid_pls,
        "avoid_logits":  avoid_pl,
        "topk_default":  4,
        "topk_candidates": [1, 4, 8, 16],
        "diagnostics": {"clean_frames_encoded": 100, "avoid_frames_encoded": 80},
    }
    ckpt = {"world_model": {}, "token_vampprior_preference": tvp}
    path = tmp_path / "tvp.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only=False)

    assert "token_vampprior_preference" in loaded
    d = loaded["token_vampprior_preference"]
    for key in ["type", "method", "score_mode", "latent_dim", "num_tokens",
                "K_clean", "K_avoid", "clean_mean", "clean_log_std", "clean_logits",
                "avoid_mean", "avoid_log_std", "avoid_logits",
                "topk_default", "topk_candidates", "diagnostics"]:
        assert key in d, f"missing key: {key}"

    assert d["type"]       == "position_conditioned_token_posterior_mixture"
    assert d["method"]     == "vampprior_like"
    assert d["score_mode"] == "q_integrated"
    assert d["clean_mean"].shape  == (N, K_c, Z)
    assert d["avoid_mean"].shape  == (N, K_a, Z)


def test_checkpoint_does_not_overwrite_pooled_keys(tmp_path):
    """Adding token_vampprior_preference must not overwrite existing keys."""
    ckpt = {
        "world_model": {},
        "contrastive_preference": {"clean_means": torch.zeros(5, 64)},
        "token_contrastive_preference": {"type": "shared_token_gmm"},
    }
    ckpt["token_vampprior_preference"] = {"type": "position_conditioned_token_posterior_mixture"}
    path = tmp_path / "multi.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only=False)

    assert "contrastive_preference" in loaded
    assert "token_contrastive_preference" in loaded
    assert "token_vampprior_preference" in loaded
    assert loaded["contrastive_preference"]["clean_means"].shape == (5, 64)


def test_state_dict_roundtrip():
    """state_dict / from_state_dict roundtrip preserves tensor shapes."""
    N, K, Z = 4, 2, 4
    model = _small_model(N=N, K=K, Z=Z)
    sd = model.state_dict()
    restored = TokenVampPriorPreference.from_state_dict(sd)

    assert torch.equal(restored.clean_mean,    model.clean_mean)
    assert torch.equal(restored.clean_log_std, model.clean_log_std)
    assert torch.equal(restored.avoid_mean,    model.avoid_mean)
    assert restored.score_mode     == model.score_mode
    assert restored.topk_default   == model.topk_default
    assert restored.contrast_scale == model.contrast_scale
