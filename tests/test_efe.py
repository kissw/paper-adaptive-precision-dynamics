import torch
from active_inference.planning.efe import EFEScorer
from active_inference.models.ensemble import EnsembleTransitionHeads
from active_inference.training.preference import PreferenceModel


def test_efe_lower_near_gmm():
    torch.manual_seed(0)
    pref = PreferenceModel(K=2, latent_dim=8)
    with torch.no_grad():
        pref.means.data = torch.tensor([[3.0]*8, [-3.0]*8])
        pref.log_stds.data = torch.zeros(2, 8) - 1.0
    ens = EnsembleTransitionHeads(feat_dim=72, stoch_dim=8, hidden_dim=32, num_heads=3)
    scorer = EFEScorer(beta_instrumental=1.0, beta_epistemic=0.0)

    near_mean = torch.tensor([[3.0]*8])
    near_std = torch.ones(1, 8) * 0.5
    far_mean = torch.tensor([[10.0]*8])
    far_std = torch.ones(1, 8) * 0.5
    near_feat = torch.randn(1, 72)
    far_feat = torch.randn(1, 72)

    efe_near = scorer.score(
        [near_feat], [near_mean], [near_std], pref, ens
    )
    efe_far = scorer.score(
        [far_feat], [far_mean], [far_std], pref, ens
    )
    assert efe_near.item() < efe_far.item()


def test_epistemic_higher_uncertain():
    ens = EnsembleTransitionHeads(feat_dim=32, stoch_dim=8, hidden_dim=16, num_heads=5)
    feat = torch.randn(4, 32)
    scorer = EFEScorer()
    unc = scorer.epistemic_value_ensemble(ens, feat)
    assert unc.shape == (4,)
    assert (unc > 0).all()


def test_efe_gradient_flows():
    torch.manual_seed(0)
    pref = PreferenceModel(K=2, latent_dim=8)
    ens = EnsembleTransitionHeads(feat_dim=72, stoch_dim=8, hidden_dim=16, num_heads=3)
    scorer = EFEScorer()

    mean = torch.randn(2, 8, requires_grad=True)
    std = torch.ones(2, 8) * 0.5
    feat = torch.randn(2, 72)

    efe = scorer.score([feat], [mean], [std], pref, ens)
    efe.sum().backward()
    assert mean.grad is not None
    assert mean.grad.abs().sum() > 0


def test_score_shape():
    pref = PreferenceModel(K=2, latent_dim=8)
    ens = EnsembleTransitionHeads(feat_dim=72, stoch_dim=8, hidden_dim=16, num_heads=3)
    scorer = EFEScorer()
    B = 6
    horizon = 5
    feats = [torch.randn(B, 72) for _ in range(horizon)]
    means = [torch.randn(B, 8) for _ in range(horizon)]
    stds = [torch.ones(B, 8) * 0.5 for _ in range(horizon)]
    result = scorer.score(feats, means, stds, pref, ens)
    assert result.shape == (B,)
