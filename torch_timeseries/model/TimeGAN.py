"""TimeGAN: generative adversarial network for time series (Yoon et al., 2019)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _GRUNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 n_layers: int, activation=None):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(x)
        out = self.fc(out)
        return self.act(out) if self.act is not None else out


class TimeGAN(nn.Module):
    """Five-network GAN for time series.

    Args:
        seq_len: window length
        n_features: number of channels
        hidden_dim: GRU hidden size (default 24)
        n_layers: GRU depth (default 3)
        gamma: supervised loss weight (default 1.0)
    """

    def __init__(self, seq_len: int, n_features: int,
                 hidden_dim: int = 24, n_layers: int = 3, gamma: float = 1.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        sig = nn.Sigmoid()
        self.embedder      = _GRUNet(n_features, hidden_dim, hidden_dim, n_layers,     sig)
        self.recovery      = _GRUNet(hidden_dim, hidden_dim, n_features, n_layers,     sig)
        self.generator     = _GRUNet(n_features, hidden_dim, hidden_dim, n_layers,     sig)
        self.supervisor    = _GRUNet(hidden_dim, hidden_dim, hidden_dim, n_layers - 1, sig)
        self.discriminator = _GRUNet(hidden_dim, hidden_dim, 1,          n_layers)

    def embed(self, x: Tensor) -> Tensor:               return self.embedder(x)
    def recover(self, h: Tensor) -> Tensor:             return self.recovery(h)
    def generate_latent(self, z: Tensor) -> Tensor:     return self.generator(z)
    def supervise(self, h: Tensor) -> Tensor:           return self.supervisor(h)
    def discriminate(self, h: Tensor) -> Tensor:        return self.discriminator(h).squeeze(-1)

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> Tensor:
        """Return (n, seq_len, n_features) on CPU."""
        z = torch.randn(n, self.seq_len, self.n_features, device=device)
        e_hat = self.generate_latent(z)
        h_hat = self.supervise(e_hat)
        return self.recovery(h_hat).cpu()
