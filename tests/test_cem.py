import torch
from active_inference.planning.cem_planner import colored_noise, iCEMPlanner
from active_inference.models.rssm import RSSM
from active_inference.models.ensemble import EnsembleTransitionHeads
from active_inference.training.preference import PreferenceModel
from active_inference.planning.efe import EFEScorer


def _make_small_components():
    rssm = RSSM(stoch_dim=8, deter_dim=16, embed_dim=16, hidden_dim=16)
    ens = EnsembleTransitionHeads(feat_dim=24, stoch_dim=8, hidden_dim=16, num_heads=3)
    pref = PreferenceModel(K=2, latent_dim=8)
    scorer = EFEScorer()
    return rssm, ens, pref, scorer


def test_colored_noise_shape():
    noise = colored_noise((50, 12, 2), beta=1.0)
    assert noise.shape == (50, 12, 2)
    assert not torch.isnan(noise).any()


def test_colored_noise_correlated():
    torch.manual_seed(0)
    noise = colored_noise((1000, 20, 1), beta=1.0)
    x = noise[:, :-1, 0].reshape(-1)
    y = noise[:, 1:, 0].reshape(-1)
    corr = torch.corrcoef(torch.stack([x, y]))[0, 1]
    assert corr > 0.2, f"Autocorrelation too low: {corr:.3f}"


def test_cem_bounds_respected():
    rssm, ens, pref, scorer = _make_small_components()
    planner = iCEMPlanner(action_dim=2, horizon=3, n_samples=20, n_elites=5, n_iters=2)
    state = rssm.initial(1)
    result = planner.plan(state, rssm, scorer, pref, ens)
    assert result.action.shape == (2,)
    assert (result.action >= -1.0).all() and (result.action <= 1.0).all()
    assert isinstance(result.efe_score, float)
    assert isinstance(result.epistemic_score, float)


def test_cem_warm_start():
    rssm, ens, pref, scorer = _make_small_components()
    planner = iCEMPlanner(action_dim=2, horizon=3, n_samples=20, n_elites=5, n_iters=2, warm_start=True)
    state = rssm.initial(1)
    _ = planner.plan(state, rssm, scorer, pref, ens)
    assert planner._prev_mean is not None
    _ = planner.plan(state, rssm, scorer, pref, ens)
    assert planner._prev_mean is not None


def test_cem_reset():
    planner = iCEMPlanner()
    planner._prev_mean = torch.randn(12, 2)
    planner.reset()
    assert planner._prev_mean is None


def test_cem_output_shape():
    rssm, ens, pref, scorer = _make_small_components()
    planner = iCEMPlanner(action_dim=2, horizon=3, n_samples=10, n_elites=3, n_iters=2)
    state = rssm.initial(1)
    result = planner.plan(state, rssm, scorer, pref, ens)
    assert result.action.shape == (2,)
