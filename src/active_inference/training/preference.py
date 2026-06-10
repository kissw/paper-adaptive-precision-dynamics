import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


class PreferenceModel(nn.Module):
    def __init__(self, K: int = 3, latent_dim: int = 64, min_std: float = 0.01):
        super().__init__()
        self._K = K
        self._latent_dim = latent_dim
        self._min_std = min_std
        self.means = nn.Parameter(torch.randn(K, latent_dim) * 0.1)
        self.log_stds = nn.Parameter(torch.zeros(K, latent_dim))
        self.logits = nn.Parameter(torch.zeros(K))

    def distribution(self) -> MixtureSameFamily:
        stds = self.log_stds.exp().clamp(min=self._min_std)
        comp = Independent(Normal(self.means, stds), 1)
        mix = Categorical(logits=self.logits)
        return MixtureSameFamily(mix, comp)

    def log_prob(self, z: Tensor) -> Tensor:
        # z: [B, latent_dim] or [S, B, latent_dim]
        return self.distribution().log_prob(z)

    @torch.no_grad()
    def _kmeans_init(self, data: Tensor):
        # k-means++ initialization
        indices = [torch.randint(data.shape[0], (1,)).item()]
        for _ in range(self._K - 1):
            dists = torch.cdist(data, data[indices])  # [N, k]
            min_dists = dists.min(dim=1).values  # [N]
            probs = min_dists / min_dists.sum()
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)
        self.means.data.copy_(data[indices])

    def update_from_latents(
        self,
        latents: Tensor,
        n_iters: int = 200,
        lr: float = 0.01,
        chunk_size: int = 50000,
    ):
        """Fit GMM via gradient descent with chunked mini-batch accumulation.

        Processes latents in chunks of chunk_size so memory usage is bounded
        regardless of the total number of tokens (e.g. 640K from TokenViT).
        Each chunk contributes a weighted fraction of the total loss so the
        gradient is equivalent to the full-batch gradient.
        """
        self._kmeans_init(latents)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        N = len(latents)
        prev_loss = float("inf")
        for i in range(n_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            for s in range(0, N, chunk_size):
                batch = latents[s : s + chunk_size]
                # Weight by chunk fraction so ∑ weighted_loss == full-batch loss
                loss = -self.log_prob(batch).mean() * (len(batch) / N)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()
            if abs(prev_loss - total_loss) < 1e-4:
                break
            prev_loss = total_loss

    def save_state(self, path: str):
        torch.save(
            {
                "means": self.means.data,
                "log_stds": self.log_stds.data,
                "logits": self.logits.data,
            },
            path,
        )

    def load_state(self, path: str):
        state = torch.load(path, weights_only=True)
        self.means.data.copy_(state["means"])
        self.log_stds.data.copy_(state["log_stds"])
        self.logits.data.copy_(state["logits"])


class ContrastivePreferenceModel(nn.Module):
    """Contrastive preference via log-ratio of two GMMs.

    Computes log p_clean(z) - log p_avoid(z) as the preference signal.
    States that look like obstacle-free driving get high scores;
    states that look like obstacle-present driving get low scores.

    In AIF terms: this encodes a prior preference that contrasts
    "what I expect to see" (clean road) against "what I want to avoid"
    (obstacle ahead). The log-ratio naturally separates the two
    distributions even when a single GMM cannot.
    """

    def __init__(
        self,
        K_clean: int = 5,
        K_avoid: int = 5,
        latent_dim: int = 64,
        min_std: float = 0.01,
        contrast_scale: float = 1.0,
    ):
        super().__init__()
        self.clean = PreferenceModel(
            K=K_clean, latent_dim=latent_dim, min_std=min_std,
        )
        self.avoid = PreferenceModel(
            K=K_avoid, latent_dim=latent_dim, min_std=min_std,
        )
        self._contrast_scale = contrast_scale

    def log_prob(self, z: Tensor) -> Tensor:
        """Log-ratio preference: positive for clean, negative for obstacle.

        Returns log p_clean(z) - contrast_scale * log p_avoid(z).
        Higher = more preferred (obstacle-free).
        """
        return (
            self.clean.log_prob(z)
            - self._contrast_scale * self.avoid.log_prob(z)
        )

    def fit(
        self,
        clean_latents: Tensor,
        avoid_latents: Tensor,
        n_iters: int = 300,
        lr: float = 0.01,
    ):
        """Fit both GMMs independently on their respective latents."""
        print(f"Fitting clean GMM (K={self.clean._K}) on "
              f"{clean_latents.shape[0]} latents...")
        self.clean.update_from_latents(clean_latents, n_iters, lr)

        print(f"Fitting avoid GMM (K={self.avoid._K}) on "
              f"{avoid_latents.shape[0]} latents...")
        self.avoid.update_from_latents(avoid_latents, n_iters, lr)

        # Diagnostic: check separation
        with torch.no_grad():
            clean_lp_clean = self.clean.log_prob(clean_latents).mean()
            clean_lp_avoid = self.avoid.log_prob(clean_latents).mean()
            avoid_lp_clean = self.clean.log_prob(avoid_latents).mean()
            avoid_lp_avoid = self.avoid.log_prob(avoid_latents).mean()
            ratio_clean = (clean_lp_clean - clean_lp_avoid).item()
            ratio_avoid = (avoid_lp_clean - avoid_lp_avoid).item()
            print(f"\nContrastive preference diagnostic:")
            print(f"  Clean latents:    log-ratio = {ratio_clean:+.2f} "
                  f"(should be positive)")
            print(f"  Obstacle latents: log-ratio = {ratio_avoid:+.2f} "
                  f"(should be negative)")
            print(f"  Separation gap:   {ratio_clean - ratio_avoid:.2f} "
                  f"nats")
