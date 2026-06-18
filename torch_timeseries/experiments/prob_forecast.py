"""Probabilistic forecasting experiment.

Differences from point forecasting (``ForecastExp``):

* **Training** — the model computes its own loss (e.g. a diffusion / ELBO
  objective); subclasses implement ``_process_train_batch(batch) -> loss``
  instead of returning (pred, true) pairs.
* **Inference** — the model returns an ensemble per window; subclasses
  implement ``_process_val_batch(batch) -> (preds, truths)`` with ``preds``
  shaped ``(B, pred_len, num_features, num_samples)`` and ``truths``
  shaped ``(B, pred_len, num_features)``.
* **Evaluation windows** — sampling many trajectories per window is
  expensive, so val/test default to non-overlapping windows
  (``fast_val`` / ``fast_test``) while training keeps the sliding window.
* **Metrics** — CRPS, CRPS-sum, QICE, PICP plus point metrics on the
  ensemble mean; early stopping monitors val CRPS.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from torch_timeseries.scaler import *  # noqa: F401,F403 — scaler registry for parse_type
from ..dataloader.v2 import (
    ForecastDataModule,
    LoaderConfig,
    SplitConfig,
    TimeEncConfig,
    TSBatch,
    WindowConfig,
)
from ..metrics import CRPS, CRPSSum, PICP, QICE, ProbMAE, ProbMSE, ProbRMSE
from .forecast import ForecastExp


@dataclass
class ProbForecastExp(ForecastExp):
    num_samples: int = 100
    """Ensemble size S drawn per window at inference time."""
    fast_val: bool = True
    fast_test: bool = True
    """Evaluate on non-overlapping windows (the default) to keep sampling
    inference tractable; set False for the dense sliding-window protocol."""

    @property
    def monitor_metric(self) -> str:
        return "crps"

    def _init_metrics(self):
        # Sample-based metrics are cheap on CPU and avoid device round-trips
        # for the (B, O, N, S) ensembles.
        self.metrics = MetricCollection(
            metrics={
                "crps": CRPS(),
                "crps_sum": CRPSSum(),
                "qice": QICE(),
                "picp": PICP(),
                "mse": ProbMSE(),
                "mae": ProbMAE(),
                "rmse": ProbRMSE(),
            }
        )
        self.metrics.to("cpu")

    def _init_data_loader(self):
        self._init_dataset()
        from ..utils.parse_type import parse_type
        self.scaler = parse_type(self.scaler_type, globals=globals())()

        window_cfg = WindowConfig(
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            fast_val=self.fast_val,
            fast_test=self.fast_test,
            time_enc_cfg=TimeEncConfig(
                time_enc=self.time_enc,
                freq=getattr(self.dataset, "freq", None),
            ),
            input_columns=self.input_columns or None,
            target_columns=self.target_columns or None,
        )
        if self.train_ratio is None and self.test_ratio is None:
            split_cfg = None  # dataset default (canonical borders or 7:1:2)
        else:
            split_cfg = SplitConfig(
                train=self.train_ratio if self.train_ratio is not None else 0.7,
                test=self.test_ratio,
            )
        self.datamodule = ForecastDataModule(
            dataset=self.dataset,
            scaler=self.scaler,
            window=window_cfg,
            split=split_cfg,
            loader=LoaderConfig(
                batch_size=self.batch_size,
                num_workers=self.num_worker,
                shuffle_train=True,
            ),
        )
        self.train_loader = self.datamodule.train_loader
        self.val_loader = self.datamodule.val_loader
        self.test_loader = self.datamodule.test_loader
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps:   {self.val_steps} (fast_val={self.fast_val})")
        print(f"test steps:  {self.test_steps} (fast_test={self.fast_test})")

    # ------------------------------------------------------------------ #
    # model construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_model(self):
        """Build and return the model.  Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _build_model()")

    def _init_model(self):
        self.model = self._build_model().to(self.device)

    # ------------------------------------------------------------------ #
    # subclass contract                                                  #
    # ------------------------------------------------------------------ #

    def _process_train_batch(self, batch: TSBatch) -> torch.Tensor:
        """Compute the training loss for one batch.

        Probabilistic models usually optimize a likelihood-style objective
        (ELBO, diffusion loss, ...) rather than an (x, y) regression pair, so
        the model returns the loss directly.
        """
        raise NotImplementedError()

    def _process_val_batch(self, batch: TSBatch):
        """Draw an ensemble for one batch.

        Returns:
            preds:  (B, pred_len, num_features, num_samples)
            truths: (B, pred_len, num_features)
        """
        raise NotImplementedError()

    # ------------------------------------------------------------------ #
    # training / evaluation loops                                        #
    # ------------------------------------------------------------------ #

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for batch in self.train_loader:
                self.model_optim.zero_grad()
                loss = self._process_train_batch(batch)
                loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                progress_bar.update(batch.x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
            return train_loss

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for batch in dataloader:
                    preds, truths = self._process_val_batch(batch)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch.y_raw.to(preds.device)
                    self.metrics.update(
                        preds.detach().cpu().contiguous(),
                        truths.detach().cpu().contiguous(),
                    )
                    progress_bar.update(batch.x.size(0))
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}
