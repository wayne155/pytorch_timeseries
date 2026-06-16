"""Evaluation metrics for time series generation quality."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_np(x: Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


# ── discriminative score ─────────────────────────────────────────────────────

class _DiscLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


def discriminative_score(
    real: Tensor,
    fake: Tensor,
    n_runs: int = 3,
    epochs: int = 5,
) -> float:
    """Mean |accuracy - 0.5| over n_runs train/test splits. Lower = better."""
    rn, fn = _to_np(real), _to_np(fake)
    N = min(len(rn), len(fn))
    rn, fn = rn[:N], fn[:N]

    rng = np.random.default_rng(42)
    scores = []

    for _ in range(n_runs):
        idx = rng.permutation(N)
        split = int(0.7 * N)
        tr, te = idx[:split], idx[split:]

        x_tr = np.concatenate([rn[tr], fn[tr]], axis=0)
        y_tr = np.array([1.0] * len(tr) + [0.0] * len(tr), dtype=np.float32)
        x_te = np.concatenate([rn[te], fn[te]], axis=0)
        y_te = np.array([1.0] * len(te) + [0.0] * len(te), dtype=np.float32)

        model = _DiscLSTM(rn.shape[2])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
            batch_size=min(64, len(x_tr)), shuffle=True,
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(torch.tensor(x_te))) > 0.5).float()
            acc = (preds == torch.tensor(y_te)).float().mean().item()
        scores.append(abs(acc - 0.5))

    return float(np.mean(scores))


# ── predictive score ─────────────────────────────────────────────────────────

class _PredGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, n_features)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


def predictive_score(
    real: Tensor,
    fake: Tensor,
    epochs: int = 5,
) -> float:
    """Train GRU on synthetic, measure MAE on real. Lower = better."""
    rn, fn = _to_np(real), _to_np(fake)
    C = rn.shape[2]

    x_syn = torch.tensor(fn[:, :-1, :], dtype=torch.float32)
    y_syn = torch.tensor(fn[:, 1:, :],  dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_syn, y_syn), batch_size=min(64, len(x_syn)), shuffle=True)

    model = _PredGRU(C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    x_real = torch.tensor(rn[:, :-1, :], dtype=torch.float32)
    y_real = torch.tensor(rn[:, 1:, :],  dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(x_real), y_real).item())


# ── Context-FID ───────────────────────────────────────────────────────────────

class _GRUEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        _, h = self.gru(x)
        return h[-1]  # (B, hidden)


def _frechet(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    cov = sqrtm(s1 @ s2)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(max(0.0, diff @ diff + np.trace(s1 + s2 - 2.0 * cov)))


def context_fid(real: Tensor, fake: Tensor) -> float:
    """Frechet distance in GRU embedding space. Lower = better."""
    rn, fn = _to_np(real), _to_np(fake)
    C = rn.shape[2]

    # Quick unsupervised pre-training: predict last timestep from full sequence
    x_t = torch.tensor(rn, dtype=torch.float32)
    encoder = _GRUEncoder(C)
    head = nn.Linear(64, C)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_t), batch_size=min(64, len(x_t)), shuffle=True)

    encoder.train(); head.train()
    for _ in range(3):
        for (xb,) in loader:
            opt.zero_grad()
            loss_fn(head(encoder(xb)), xb[:, -1, :]).backward()
            opt.step()

    encoder.eval()
    with torch.no_grad():
        emb_r = encoder(torch.tensor(rn, dtype=torch.float32)).numpy()
        emb_f = encoder(torch.tensor(fn, dtype=torch.float32)).numpy()

    eps = 1e-6 * np.eye(emb_r.shape[1])
    mu_r, s_r = emb_r.mean(0), np.cov(emb_r, rowvar=False) + eps
    mu_f, s_f = emb_f.mean(0), np.cov(emb_f, rowvar=False) + eps
    return _frechet(mu_r, s_r, mu_f, s_f)


# ── correlational score ───────────────────────────────────────────────────────

def correlational_score(real: Tensor, fake: Tensor, max_lag: int = 20) -> float:
    """MSE between per-feature autocorrelation vectors. Lower = better."""
    rn, fn = _to_np(real), _to_np(fake)
    T = rn.shape[1]
    effective_lag = min(max_lag, T - 1)

    def _ac(x: np.ndarray) -> np.ndarray:
        # x: (N, T, C) -> (effective_lag, C)
        result = np.zeros((effective_lag, x.shape[2]))
        for lag in range(1, effective_lag + 1):
            a, b = x[:, :-lag, :], x[:, lag:, :]
            cov = (a * b).mean(axis=(0, 1))
            std = a.std(axis=(0, 1)) * b.std(axis=(0, 1)) + 1e-8
            result[lag - 1] = cov / std
        return result

    return float(np.mean((_ac(rn) - _ac(fn)) ** 2))
