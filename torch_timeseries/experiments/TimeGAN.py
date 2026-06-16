"""TimeGAN generation experiment (multi-phase training)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from ..model.TimeGAN import TimeGAN
from ..dataloader.v2.batch import TSBatch
from ..metrics.generation import (
    discriminative_score, predictive_score, context_fid, correlational_score,
)
from ..utils.reproduce import reproducible
from .generation import GenerationExp


@dataclass
class TimeGANGeneration(GenerationExp):
    model_type: str = "TimeGAN"
    hidden_dim: int = 24
    n_layers: int = 3
    gamma: float = 1.0
    epochs_ae: int = 200
    epochs_sup: int = 200
    epochs_joint: int = 50

    def _init_model(self) -> None:
        self.model = TimeGAN(
            seq_len=self.seq_len,
            n_features=self.num_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            gamma=self.gamma,
        ).to(self.device)

    def generate(self, n_samples: int, condition=None) -> torch.Tensor:
        self.model.eval()
        return self.model.generate(n_samples, device=self.device)

    # ── phase losses ──────────────────────────────────────────────────────────

    def _ae_loss(self, x: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(self.model.recover(self.model.embed(x)), x)

    def _sup_loss(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model.embed(x)
        return nn.MSELoss()(self.model.supervise(h)[:, :-1, :], h[:, 1:, :])

    def _gen_loss(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(x)
        e_hat = self.model.generate_latent(z)
        h_hat = self.model.supervise(e_hat)
        g_u = nn.BCEWithLogitsLoss()(
            self.model.discriminate(h_hat),
            torch.ones(x.shape[0], x.shape[1], device=x.device),
        )
        g_s = nn.MSELoss()(h_hat[:, :-1, :], e_hat[:, 1:, :])
        x_hat = self.model.recover(self.model.embed(x))
        g_v = (torch.abs(x_hat.std(0) - x.std(0)).mean() +
               torch.abs(x_hat.mean(0) - x.mean(0)).mean())
        return g_u + self.model.gamma * g_s + g_v

    def _disc_loss(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model.embed(x)
        B, T = x.shape[0], x.shape[1]
        y_real = self.model.discriminate(h)
        z = torch.randn_like(x)
        h_fake = self.model.supervise(self.model.generate_latent(z))
        y_fake = self.model.discriminate(h_fake)
        ones = torch.ones(B, T, device=x.device)
        zeros = torch.zeros(B, T, device=x.device)
        return (nn.BCEWithLogitsLoss()(y_real, ones) +
                nn.BCEWithLogitsLoss()(y_fake, zeros))

    def _make_opt(self, *modules):
        params = [p for m in modules for p in m.parameters()]
        return torch.optim.Adam(params, lr=self.lr)

    # ── override run() for multi-phase training ───────────────────────────────

    def run(self, seed: int = 0) -> dict:
        reproducible(seed)
        self._init_data_loader()
        self._init_model()

        m = self.model
        opt_ae  = self._make_opt(m.embedder, m.recovery)
        opt_sup = self._make_opt(m.generator, m.supervisor)
        opt_g   = self._make_opt(m.generator, m.supervisor, m.embedder, m.recovery)
        opt_d   = self._make_opt(m.discriminator)

        def _loop(loss_fn, opt, n_epochs, desc):
            for _ in tqdm(range(n_epochs), desc=desc, leave=False):
                m.train()
                for batch in self.train_loader:
                    x = batch.x.to(self.device)
                    opt.zero_grad()
                    loss_fn(x).backward()
                    opt.step()

        _loop(self._ae_loss,  opt_ae,  self.epochs_ae,  "Phase1 AE")
        _loop(self._sup_loss, opt_sup, self.epochs_sup, "Phase2 Sup")

        for _ in tqdm(range(self.epochs_joint), desc="Phase3 Joint", leave=False):
            m.train()
            for batch in self.train_loader:
                x = batch.x.to(self.device)
                opt_g.zero_grad()
                (self._gen_loss(x) + self._sup_loss(x)).backward()
                opt_g.step()
                opt_d.zero_grad()
                d_loss = self._disc_loss(x)
                if d_loss.item() > 0.15:
                    d_loss.backward()
                    opt_d.step()

        return self._evaluate()
