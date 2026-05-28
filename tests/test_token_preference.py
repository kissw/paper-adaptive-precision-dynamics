"""Tests for token-wise contrastive preference module.

Tests 1-2: extract_token_features shape contract + RSSM rejection.
Test 3:    topk_mean_token_score correctness.
Test 4:    TokenContrastivePreference GMM fit on synthetic separable data.
Test 5:    checkpoint save/load key structure.
"""

import pytest
import torch
from active_inference.models.rssm import RSSMState
from active_inference.models.token_vit import TokenRSSMState
from active_inference.training.token_preference import (
    TokenContrastivePreference,
    extract_token_features,
    topk_mean_token_score,
    compute_position_stats,
    apply_position_normalization,
)


# ---------------------------------------------------------------------------
# Test 1: extract_token_features — shape contract
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


def test_extract_deter_stoch_shape():
    B, N, D, Z = 2, 64, 256, 64
    state = _make_token_state(B, N, D, Z)
    feat = extract_token_features(state, feature_type="deter_stoch")
    assert feat.shape == (B, N, D + Z), f"expected ({B},{N},{D+Z}), got {feat.shape}"


def test_extract_deter_token_mean_shape():
    B, N, D, Z = 2, 64, 256, 64
    state = _make_token_state(B, N, D, Z)
    feat = extract_token_features(state, feature_type="deter_token_mean")
    assert feat.shape == (B, N, D + Z)


def test_extract_token_mean_shape():
    B, N, D, Z = 2, 64, 256, 64
    state = _make_token_state(B, N, D, Z)
    feat = extract_token_features(state, feature_type="token_mean")
    assert feat.shape == (B, N, Z)


# ---------------------------------------------------------------------------
# Test 2: extract_token_features — reject RSSM 2D state
# ---------------------------------------------------------------------------

def test_extract_rejects_rssm_2d_state():
    """RSSM state has 2D deter — must raise ValueError with clear message."""
    rssm_state = RSSMState(
        deter=torch.zeros(2, 256),
        stoch=torch.zeros(2, 64),
        mean=torch.zeros(2, 64),
        std=torch.ones(2, 64),
    )
    with pytest.raises(ValueError, match="3D"):
        extract_token_features(rssm_state)


# ---------------------------------------------------------------------------
# Test 3: topk_mean_token_score correctness
# ---------------------------------------------------------------------------

def test_topk_k1_equals_max():
    scores = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.5, 4.0]])  # (2, 3)
    result = topk_mean_token_score(scores, k=1)
    expected = scores.max(dim=-1).values
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_topk_kN_equals_mean():
    scores = torch.randn(4, 16)
    result = topk_mean_token_score(scores, k=16)
    expected = scores.mean(dim=-1)
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"


def test_topk_shape():
    for B, N, k in [(3, 64, 4), (1, 64, 8), (5, 64, 1)]:
        scores = torch.randn(B, N)
        result = topk_mean_token_score(scores, k=k)
        assert result.shape == (B,), f"k={k}: expected ({B},), got {result.shape}"


def test_topk_clips_k_to_N():
    scores = torch.randn(2, 8)
    # k=100 >> N=8 — should not error, clips to N
    result = topk_mean_token_score(scores, k=100)
    assert result.shape == (2,)
    assert torch.allclose(result, scores.mean(dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: TokenContrastivePreference GMM fit — synthetic separable data
# ---------------------------------------------------------------------------

def test_token_contrastive_separation():
    """Clean tokens near +1, obstacle tokens near -1 — expect positive gap."""
    torch.manual_seed(0)
    D = 4  # small dim for fast fitting
    N_samples = 400

    # Clean: cluster around +1 in first dimension
    clean_flat = torch.randn(N_samples, D) * 0.3
    clean_flat[:, 0] += 1.0

    # Avoid: cluster around -1 in first dimension
    avoid_flat = torch.randn(N_samples, D) * 0.3
    avoid_flat[:, 0] -= 1.0

    model = TokenContrastivePreference(
        K_clean=2, K_avoid=2, token_dim=D, min_std=0.01,
    )
    model.fit(clean_flat, avoid_flat, n_iters=200, lr=0.01)

    with torch.no_grad():
        c_score = model.log_prob_flat(clean_flat).mean().item()
        a_score = model.log_prob_flat(avoid_flat).mean().item()
        gap = c_score - a_score

    assert c_score > 0, f"clean log-ratio should be positive, got {c_score:.3f}"
    assert a_score < 0, f"avoid log-ratio should be negative, got {a_score:.3f}"
    assert gap > 0,     f"gap should be positive, got {gap:.3f}"


def test_score_frames_shape():
    """score_frames aggregates (B, N, D) -> (B,)."""
    torch.manual_seed(1)
    D, N, B = 4, 8, 3
    model = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=D)
    clean_flat = torch.randn(200, D) + 1.0
    avoid_flat = torch.randn(200, D) - 1.0
    model.fit(clean_flat, avoid_flat, n_iters=50, lr=0.01)

    z_tokens = torch.randn(B, N, D)
    for k in [1, 4, N]:
        result = model.score_frames(z_tokens, topk=k)
        assert result.shape == (B,), f"k={k}: expected ({B},), got {result.shape}"


def test_token_scores_shape():
    """token_scores returns (B, N) without aggregation."""
    D, N, B = 4, 8, 2
    model = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=D)
    # No fitting needed for shape check
    z_tokens = torch.randn(B, N, D)
    scores = model.token_scores(z_tokens)
    assert scores.shape == (B, N), f"expected ({B},{N}), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 5: Checkpoint key structure
# ---------------------------------------------------------------------------

def test_checkpoint_key_structure(tmp_path):
    """token_contrastive_preference key exists with expected subkeys."""
    D, K = 4, 2
    model = TokenContrastivePreference(K_clean=K, K_avoid=K, token_dim=D)

    ckpt = {
        "world_model": {},  # mock world model weights
        "token_contrastive_preference": {
            "type":          "shared_token_gmm",
            "feature_type":  "deter_stoch",
            "token_dim":     D,
            "num_tokens":    64,
            "K_clean":       K,
            "K_avoid":       K,
            "covariance_type": "diag",
            "contrast_scale": 1.0,
            "topk_default":   4,
            "topk_candidates": [1, 4, 8, 16],
            "clean_means":    model.clean.means.data.cpu(),
            "clean_log_stds": model.clean.log_stds.data.cpu(),
            "clean_logits":   model.clean.logits.data.cpu(),
            "avoid_means":    model.avoid.means.data.cpu(),
            "avoid_log_stds": model.avoid.log_stds.data.cpu(),
            "avoid_logits":   model.avoid.logits.data.cpu(),
            "diagnostics": {
                "clean_frames_encoded": 100,
                "avoid_frames_encoded": 100,
            },
        },
    }

    path = tmp_path / "test_ckpt.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only=False)

    assert "token_contrastive_preference" in loaded, "key missing"
    tcp = loaded["token_contrastive_preference"]

    assert tcp["type"]          == "shared_token_gmm"
    assert tcp["feature_type"]  == "deter_stoch"
    assert tcp["token_dim"]     == D
    assert tcp["K_clean"]       == K
    assert tcp["K_avoid"]       == K
    assert tcp["topk_default"]  == 4
    assert "clean_means"    in tcp
    assert "avoid_means"    in tcp
    assert "diagnostics"    in tcp

    # Pooled contrastive_preference key is absent (separate concern)
    assert "contrastive_preference" not in loaded


def test_existing_pooled_key_preserved(tmp_path):
    """Adding token GMM key must not overwrite existing pooled contrastive_preference."""
    ckpt = {
        "world_model": {},
        "contrastive_preference": {"clean_means": torch.zeros(5, 64)},
    }
    ckpt["token_contrastive_preference"] = {"type": "shared_token_gmm"}

    path = tmp_path / "test_both.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only=False)

    assert "contrastive_preference" in loaded, "pooled key was lost"
    assert "token_contrastive_preference" in loaded, "token key missing"
    # Values in pooled key unchanged
    assert loaded["contrastive_preference"]["clean_means"].shape == (5, 64)


# ---------------------------------------------------------------------------
# Test 6: compute_position_stats
# ---------------------------------------------------------------------------

def test_compute_position_stats_shape():
    """Output shapes are (N_tokens, token_dim) for both mu and std."""
    torch.manual_seed(0)
    F, N, D = 100, 64, 320
    tokens = torch.randn(F, N, D)
    mu_pos, std_pos = compute_position_stats(tokens, eps=1e-6)
    assert mu_pos.shape  == (N, D), f"mu_pos shape wrong: {mu_pos.shape}"
    assert std_pos.shape == (N, D), f"std_pos shape wrong: {std_pos.shape}"


def test_compute_position_stats_std_positive():
    """std_pos must be strictly positive (clamped to eps)."""
    torch.manual_seed(1)
    F, N, D = 50, 64, 320
    tokens = torch.randn(F, N, D)
    _, std_pos = compute_position_stats(tokens, eps=1e-6)
    assert (std_pos > 0).all(), "std_pos contains non-positive values"


def test_compute_position_stats_std_clamped_constant_input():
    """Constant token (std=0 before clamping) must yield std >= eps."""
    eps = 1e-4
    F, N, D = 10, 4, 8
    tokens = torch.ones(F, N, D)  # all identical -> raw std == 0
    _, std_pos = compute_position_stats(tokens, eps=eps)
    assert (std_pos >= eps).all(), f"clamping failed: min std = {std_pos.min().item()}"


# ---------------------------------------------------------------------------
# Test 7: apply_position_normalization
# ---------------------------------------------------------------------------

def test_apply_position_normalization_shape():
    """Output shape matches input shape."""
    F, N, D = 80, 64, 320
    tokens  = torch.randn(F, N, D)
    mu_pos  = tokens.mean(dim=0)
    std_pos = tokens.std(dim=0).clamp(min=1e-6)
    out = apply_position_normalization(tokens, mu_pos, std_pos)
    assert out.shape == (F, N, D)


def test_apply_position_normalization_zero_mean_unit_std():
    """After normalization, per-position mean ≈ 0 and std ≈ 1."""
    torch.manual_seed(2)
    F, N, D = 200, 16, 32
    # Give each position a distinct offset so position bias is non-trivial
    offset = torch.arange(N, dtype=torch.float32).unsqueeze(-1) * 2.0  # (N, 1)
    tokens = torch.randn(F, N, D) + offset.unsqueeze(0)

    mu_pos, std_pos = compute_position_stats(tokens, eps=1e-6)
    normed = apply_position_normalization(tokens, mu_pos, std_pos)

    # Per-position mean should be ≈ 0
    pos_mean = normed.mean(dim=0)   # (N, D)
    assert pos_mean.abs().max().item() < 1e-4, (
        f"mean not ≈ 0 after normalisation: max abs = {pos_mean.abs().max().item():.6f}"
    )
    # Per-position std should be ≈ 1
    pos_std = normed.std(dim=0)     # (N, D)
    assert (pos_std - 1.0).abs().max().item() < 5e-3, (
        f"std not ≈ 1 after normalisation: max dev = {(pos_std-1).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: pos_norm GMM on synthetic position-biased data
# ---------------------------------------------------------------------------

def test_pos_norm_token_gmm_separates_position_biased_data():
    """
    Clean and obstacle tokens have the same class offset but different position
    biases. After position normalization, the GMM must still achieve positive gap
    (class separation survives; position bias is removed, not class signal).
    """
    torch.manual_seed(3)
    F_c, F_a, N, D = 80, 80, 8, 4

    # Position bias added to both — must be removed by pos_norm
    pos_bias = torch.arange(N, dtype=torch.float32).unsqueeze(-1) * 3.0  # (N, 1)

    # Class offsets: clean near +1, avoid near -1 in dim 0
    clean_tokens = torch.randn(F_c, N, D) * 0.3 + pos_bias.unsqueeze(0)
    clean_tokens[:, :, 0] += 1.0

    avoid_tokens = torch.randn(F_a, N, D) * 0.3 + pos_bias.unsqueeze(0)
    avoid_tokens[:, :, 0] -= 1.0

    # Compute position stats from clean only (as in the script)
    mu_pos, std_pos = compute_position_stats(clean_tokens, eps=1e-6)
    clean_norm = apply_position_normalization(clean_tokens, mu_pos, std_pos)
    avoid_norm = apply_position_normalization(avoid_tokens, mu_pos, std_pos)

    # Flatten and fit GMM on normalised features
    clean_flat = clean_norm.reshape(-1, D)
    avoid_flat = avoid_norm.reshape(-1, D)

    model = TokenContrastivePreference(K_clean=2, K_avoid=2, token_dim=D, min_std=0.01)
    model.fit(clean_flat, avoid_flat, n_iters=200, lr=0.01)

    with torch.no_grad():
        c_score = model.log_prob_flat(clean_flat).mean().item()
        a_score = model.log_prob_flat(avoid_flat).mean().item()
        gap = c_score - a_score

    assert gap > 0, (
        f"pos_norm GMM gap not positive: clean={c_score:.3f}  avoid={a_score:.3f}  gap={gap:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 9: checkpoint structure — pos_norm and shared modes
# ---------------------------------------------------------------------------

def test_checkpoint_pos_norm_has_position_stats(tmp_path):
    """pos_norm checkpoint must contain position_stats with mu_pos and std_pos."""
    N, D = 8, 4
    mu_pos  = torch.zeros(N, D)
    std_pos = torch.ones(N, D)

    tcp = {
        "type":         "shared_token_gmm",
        "mode":         "pos_norm",
        "feature_type": "deter_stoch",
        "token_dim":    D,
        "num_tokens":   N,
        "K_clean":      2,
        "K_avoid":      2,
        "topk_default": 4,
        "topk_candidates": [1, 4, 8, 16],
        "position_stats": {
            "mu_pos":  mu_pos,
            "std_pos": std_pos,
            "eps":     1e-6,
        },
        "diagnostics": {},
    }
    path = tmp_path / "pos_norm.pt"
    torch.save({"token_contrastive_preference": tcp}, path)
    loaded = torch.load(path, weights_only=False)["token_contrastive_preference"]

    assert loaded["mode"] == "pos_norm"
    assert "position_stats" in loaded, "position_stats missing for pos_norm"
    ps = loaded["position_stats"]
    assert "mu_pos"  in ps and "std_pos" in ps and "eps" in ps
    assert ps["mu_pos"].shape  == (N, D)
    assert ps["std_pos"].shape == (N, D)


def test_checkpoint_shared_has_no_position_stats(tmp_path):
    """shared checkpoint must NOT contain position_stats."""
    tcp = {
        "type":         "shared_token_gmm",
        "mode":         "shared",
        "feature_type": "deter_stoch",
        "token_dim":    4,
        "num_tokens":   8,
        "K_clean":      2,
        "K_avoid":      2,
        "topk_default": 4,
        "topk_candidates": [1, 4, 8, 16],
        "diagnostics": {},
    }
    path = tmp_path / "shared.pt"
    torch.save({"token_contrastive_preference": tcp}, path)
    loaded = torch.load(path, weights_only=False)["token_contrastive_preference"]

    assert loaded["mode"] == "shared"
    assert "position_stats" not in loaded, "position_stats must be absent for shared mode"
