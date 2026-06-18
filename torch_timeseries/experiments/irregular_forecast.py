from __future__ import annotations
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm

from ..core import BaseIrrelevant, BaseRelevant
from ..dataloader.v2.forecast import LoaderConfig, SplitConfig
from ..dataloader.v2.irregular_forecast import (
    IrregularForecastConfig, IrregularForecastDataModule,
)
from ..dataloader.v2.irregular_batch import IrregularTSBatch
from ..scaler import StandardScaler
from ..utils.early_stop import EarlyStopping
from ..utils.reproduce import reproducible


def _get_forecast_dataset(name: str, root: str):
    from ..dataset.irregular import IrregularWrapper, PhysioNet2012, PhysioNet2019
    from ..dataset import ETTh1, ETTh2, ETTm1, ETTm2
    _regular = {
        "ETTh1": ETTh1, "ETTh2": ETTh2, "ETTm1": ETTm1, "ETTm2": ETTm2,
    }
    if name in _regular:
        return IrregularWrapper(_regular[name](root=root), drop_rate=0.3)
    _irreg = {"PhysioNet2012": PhysioNet2012, "PhysioNet2019": PhysioNet2019}
    if name in _irreg:
        return _irreg[name](root=root)
    raise ValueError(f"Unknown dataset for irregular forecast: {name!r}")


@dataclass
class IrregularForecastSettings:
    obs_frac: float = 0.7
    time_enc: int = 1
    freq: Optional[str] = None


@dataclass
class IrregularForecastExp(BaseRelevant, BaseIrrelevant, IrregularForecastSettings):
    """Base experiment for irregular time-series forecasting.

    Subclasses must implement ``_init_model()``.
    Loss: masked MSE on all future observation points.
    """

    def _init_model(self) -> None:
        raise NotImplementedError

    def _init_data_loader(self) -> None:
        if hasattr(self, "_toy_dataset"):
            dataset = self._toy_dataset
        else:
            dataset = _get_forecast_dataset(self.dataset_type, self.data_path)
        self.dm = IrregularForecastDataModule(
            dataset=dataset,
            scaler=StandardScaler(),
            window=IrregularForecastConfig(
                obs_frac=self.obs_frac,
                time_enc=self.time_enc,
                freq=self.freq,
            ),
            split=SplitConfig(train=0.7, val=0.1, test=0.2),
            loader=LoaderConfig(batch_size=self.batch_size, num_workers=self.num_worker),
        )
        self.train_loader = self.dm.train_loader
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader

    def _init_metrics(self) -> None:
        self.metrics = MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )
        self.metrics.to(self.device)

    def _init_optimizer(self) -> None:
        from torch.optim import Adam
        self.optimizer = Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.l2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

    def _setup(self) -> None:
        self._init_data_loader()
        self._init_metrics()
        self._setuped = True

    def _process_one_batch(self, batch: IrregularTSBatch):
        x = batch.x.float().to(self.device)
        t = batch.t.float().to(self.device)
        mask = batch.mask.float().to(self.device)
        t_query = batch.t_query.float().to(self.device)
        y = batch.y.float().to(self.device)
        qmask = batch.query_mask.float().to(self.device)
        pred = self.model(x, t, mask, t_query=t_query)   # (B, Tq, F)
        return pred, y, qmask

    def _masked_mse_loss(self, pred, y, qmask) -> torch.Tensor:
        loss_all = (pred - y) ** 2
        return (loss_all * qmask).sum() / (qmask.sum() + 1e-8)

    def _train(self) -> None:
        self.model.train()
        with tqdm(total=len(self.train_loader.dataset), leave=False) as pb:
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses_step = loss.item()
                pb.update(batch.x.shape[0])

    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        losses = []
        with torch.no_grad():
            for batch in loader:
                pred, y, qmask = self._process_one_batch(batch)
                loss = self._masked_mse_loss(pred, y, qmask)
                losses.append(loss.item())
                B, Tq, F = pred.shape
                flat_pred = pred[qmask.bool()].reshape(-1)
                flat_y = y[qmask.bool()].reshape(-1)
                if flat_pred.numel() > 0:
                    self.metrics.update(flat_pred, flat_y)
        result = {k: float(v.compute()) for k, v in self.metrics.items()}
        result["loss"] = float(np.mean(losses))
        return result

    def _run_identifier(self, seed: int) -> str:
        ident = asdict(self)
        ident["seed"] = seed
        return hashlib.md5(
            json.dumps(ident, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _setup_run(self, seed: int) -> None:
        if not hasattr(self, "_setuped"):
            self._setup()
        reproducible(seed)
        self._init_model()
        self._init_optimizer()
        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir, "runs", str(self.model_type),
            str(self.dataset_type), self._run_identifier(seed),
        )
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_ckpt = os.path.join(self.run_save_dir, "best_model.pth")
        self.early_stopper = EarlyStopping(self.patience, verbose=False, path=self.best_ckpt)

    def run(self, seed: int = 42) -> Dict[str, float]:
        self._setup_run(seed)
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                break
            reproducible(seed + self.current_epoch)
            self._train()
            val_result = self._evaluate(self.val_loader)
            self.current_epoch += 1
            self.early_stopper(val_result["loss"], model=self.model)
            self.scheduler.step()
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(
                torch.load(self.best_ckpt, map_location=self.device, weights_only=False))
        return self._evaluate(self.test_loader)

    def runs(self, seeds: List[int] = None) -> List[Dict[str, float]]:
        return [self.run(s) for s in (seeds or [1, 2, 3])]
