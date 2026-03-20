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

    def update_from_latents(self, latents: Tensor, n_iters: int = 200, lr: float = 0.01):
        # latents: [N, latent_dim] — detached posterior means from expert data
        self._kmeans_init(latents)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_loss = float("inf")
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = -self.log_prob(latents).mean()
            loss.backward()
            optimizer.step()
            if abs(prev_loss - loss.item()) < 1e-4:
                break
            prev_loss = loss.item()

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
