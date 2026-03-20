import torch
from active_inference.models.ensemble import EnsembleTransitionHeads


def test_ensemble_k_outputs():
    ens = EnsembleTransitionHeads(feat_dim=320, num_heads=5)
    feat = torch.randn(4, 320)
    preds = ens(feat)
    assert preds.shape == (5, 4, 64)
    assert not torch.allclose(preds[0], preds[1])


def test_epistemic_positive():
    ens = EnsembleTransitionHeads(feat_dim=320, num_heads=5)
    feat = torch.randn(4, 320)
    unc = ens.epistemic_uncertainty(feat)
    assert unc.shape == (4,)
    assert (unc > 0).all()


def test_ensemble_shape():
    ens = EnsembleTransitionHeads(feat_dim=128, stoch_dim=32, hidden_dim=64, num_heads=3)
    feat = torch.randn(8, 128)
    preds = ens(feat)
    assert preds.shape == (3, 8, 32)
    unc = ens.epistemic_uncertainty(feat)
    assert unc.shape == (8,)
