import torch
from active_inference.training.preference import PreferenceModel


def test_gmm_log_prob_higher_at_means():
    pref = PreferenceModel(K=3, latent_dim=8)
    with torch.no_grad():
        at_means = pref.log_prob(pref.means)
        at_random = pref.log_prob(torch.randn(3, 8) * 10)
    assert (at_means > at_random).all()


def test_gmm_fit_recovers_known():
    torch.manual_seed(0)
    # Create data from known 2-component mixture in 8 dims
    true_means = torch.tensor([[2.0] * 8, [-2.0] * 8])
    data = torch.cat(
        [
            true_means[0].unsqueeze(0) + torch.randn(500, 8) * 0.3,
            true_means[1].unsqueeze(0) + torch.randn(500, 8) * 0.3,
        ]
    )
    pref = PreferenceModel(K=2, latent_dim=8)
    pref.update_from_latents(data, n_iters=200, lr=0.01)
    fitted_means = pref.means.data.sort(dim=0).values
    true_sorted = true_means.sort(dim=0).values
    assert torch.allclose(fitted_means, true_sorted, atol=1.0)


def test_gmm_save_load_roundtrip(tmp_path):
    pref = PreferenceModel(K=3, latent_dim=16)
    z = torch.randn(10, 16)
    lp_before = pref.log_prob(z)
    path = str(tmp_path / "gmm.pt")
    pref.save_state(path)
    pref2 = PreferenceModel(K=3, latent_dim=16)
    pref2.load_state(path)
    lp_after = pref2.log_prob(z)
    assert torch.allclose(lp_before, lp_after)


def test_gmm_no_component_collapse():
    torch.manual_seed(42)
    data = torch.cat(
        [
            torch.randn(300, 8) + 3,
            torch.randn(300, 8) - 3,
            torch.randn(300, 8),
        ]
    )
    pref = PreferenceModel(K=3, latent_dim=8)
    pref.update_from_latents(data, n_iters=200)
    # All K means should be distinct
    means = pref.means.data
    for i in range(3):
        for j in range(i + 1, 3):
            dist = (means[i] - means[j]).norm()
            assert dist > 0.5, f"Components {i} and {j} collapsed: dist={dist:.3f}"


def test_gmm_distribution_shape():
    pref = PreferenceModel(K=3, latent_dim=64)
    z = torch.randn(10, 64)
    lp = pref.log_prob(z)
    assert lp.shape == (10,)
