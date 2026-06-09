from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict

import torch
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_timeseries.dataloader.v2 import (
    ForecastDataModule,
    LoaderConfig,
    SplitConfig,
    WindowConfig,
)
from torch_timeseries.dataset import *
from torch_timeseries.model import Crossformer, DLinear
from torch_timeseries.scaler import *
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.results.identity import (
    build_run_config,
    config_fingerprint,
    run_id_for_seed,
)

from .configs import CrossformerConfig, DLinearConfig, ForecastConfig, RuntimeConfig

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is not installed
    class tqdm:
        def __init__(self, total=None):
            self.total = total

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, amount):
            pass

        def set_postfix(self, **kwargs):
            pass


class ForecastEngine:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        task_config: ForecastConfig,
        model_config,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task = "Forecast"
        self.task_config = task_config
        self.model_config = model_config
        self.runtime = runtime_config
        self.current_epoch = 0
        self.history = None
        self.run_config = None
        self.config_hash = ""
        self.run_id = ""
        self.current_seed = None

    def _init_dataset(self) -> None:
        self.dataset = parse_type(self.dataset_name, globals())(
            root=self.runtime.data_path
        )

    def _init_datamodule(self) -> None:
        scaler = parse_type(self.runtime.scaler_type, globals())()
        self.datamodule = ForecastDataModule(
            dataset=self.dataset,
            scaler=scaler,
            window=WindowConfig(
                window=self.task_config.windows,
                horizon=self.task_config.horizon,
                steps=self.task_config.pred_len,
                stride=self.task_config.stride,
                include_raw=True,
                include_time=True,
                time_enc=self.task_config.time_enc,
                freq=self.dataset.freq,
                input_columns=self.task_config.input_columns,
                target_columns=self.task_config.target_columns,
            ),
            split=SplitConfig(
                train=self.task_config.train_ratio,
                val=self.task_config.val_ratio,
                test=self.task_config.test_ratio,
            ),
            loader=LoaderConfig(
                batch_size=self.runtime.batch_size,
                num_workers=self.runtime.num_worker,
                shuffle_train=True,
                pin_memory=self.runtime.pin_memory,
            ),
            scale_in_train=self.task_config.scale_in_train,
        )
        self.scaler = scaler
        self.train_loader = self.datamodule.train_loader
        self.val_loader = self.datamodule.val_loader
        self.test_loader = self.datamodule.test_loader

    def _build_model(self) -> None:
        raise NotImplementedError

    def _init_runtime(self) -> None:
        self.metrics = MetricCollection(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        ).to(self.runtime.device)
        self.loss_func = {"mse": MSELoss, "mae": L1Loss}[
            self.runtime.loss_func_type
        ]()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.runtime.lr,
            weight_decay=self.runtime.l2_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.runtime.epochs,
        )
        self.run_save_dir = os.path.join(
            self.runtime.save_dir,
            "runs",
            self.model_name,
            self.task,
            self.dataset_name,
            self.config_hash,
            self.run_id,
        )
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir,
            "best_model.pth",
        )
        self.early_stopper = EarlyStopping(
            self.runtime.patience,
            verbose=False,
            path=self.best_checkpoint_filepath,
        )

    def setup(self) -> None:
        self._init_dataset()
        self._init_datamodule()
        self._build_model()
        self._init_runtime()

    def _process_batch(self, batch):
        batch = batch.to(self.runtime.device)
        x = batch.x.float()
        y = batch.y.float()
        return self.model(x), y, batch.y_raw

    def _evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            for batch in loader:
                pred, true, y_raw = self._process_batch(batch)
                if self.runtime.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = y_raw.to(self.runtime.device).float()
                self.metrics.update(pred.contiguous(), true.contiguous())
        return {
            name: float(metric.compute())
            for name, metric in self.metrics.items()
        }

    def _train_epoch(self):
        self.model.train()
        losses = []
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            for batch in self.train_loader:
                batch_size = batch.x.size(0)
                self.optimizer.zero_grad()
                pred, true, y_raw = self._process_batch(batch)
                if self.runtime.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = y_raw.to(self.runtime.device).float()
                loss = self.loss_func(pred, true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.runtime.max_grad_norm,
                )
                self.optimizer.step()
                loss_value = loss.item()
                losses.append(loss_value)
                progress_bar.update(batch_size)
                progress_bar.set_postfix(
                    loss=loss_value,
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch=self.current_epoch + 1,
                    refresh=True,
                )
        return losses

    def _record_epoch(self, train_losses, val_result: Dict[str, float]) -> float:
        train_loss = (
            float(sum(train_losses) / len(train_losses))
            if train_losses
            else float("nan")
        )
        self.history["train_loss"].append(train_loss)
        self.history["val"].append(dict(val_result))
        return train_loss

    def _print_epoch_result(
        self,
        epoch: int,
        train_loss: float,
        val_result: Dict[str, float],
    ) -> None:
        parts = [
            f"Epoch {epoch + 1}/{self.runtime.epochs}",
            f"train_loss={train_loss:.6g}",
        ]
        parts.extend(
            f"val_{name}={value:.6g}"
            for name, value in sorted(val_result.items())
        )
        print(" | ".join(parts))

    def run(self, seed: int = 42) -> Dict[str, float]:
        reproducible(seed)
        self.current_seed = seed
        if not self.run_config:
            self.run_config = build_run_config(
                model=self.model_name,
                task=self.task,
                dataset=self.dataset_name,
                hparams=self.hparams(),
            )
        if not self.config_hash:
            self.config_hash = config_fingerprint(self.run_config)
        if not self.run_id:
            self.run_id = run_id_for_seed(seed, self.config_hash)
        self.setup()
        self.history = {"train_loss": [], "val": []}
        for epoch in range(self.runtime.epochs):
            self.current_epoch = epoch
            reproducible(seed + epoch)
            train_losses = self._train_epoch()
            val_result = self._evaluate(self.val_loader)
            train_loss = self._record_epoch(train_losses, val_result)
            self._print_epoch_result(epoch, train_loss, val_result)
            self.early_stopper(
                val_result[self.runtime.loss_func_type],
                model=self.model,
            )
            self.scheduler.step()
            if self.early_stopper.early_stop:
                break
        if os.path.exists(self.best_checkpoint_filepath):
            self.model.load_state_dict(
                torch.load(
                    self.best_checkpoint_filepath,
                    map_location=self.runtime.device,
                    weights_only=False,
                )
            )
        return self._evaluate(self.test_loader)

    def hparams(self) -> dict:
        out = {}
        out.update(asdict(self.task_config))
        out.update(asdict(self.model_config))
        out.update(asdict(self.runtime))
        return out

    def num_parameters(self) -> int:
        try:
            _, num_params = count_parameters(self.model)
            return num_params
        except Exception:
            return 0


class DLinearForecastEngine(ForecastEngine):
    model_config: DLinearConfig

    def _build_model(self) -> None:
        self.model = DLinear(
            seq_len=self.task_config.windows,
            pred_len=self.task_config.pred_len,
            enc_in=self.datamodule.num_features,
            individual=self.model_config.individual,
        ).to(self.runtime.device)


class CrossformerForecastEngine(ForecastEngine):
    model_config: CrossformerConfig

    def _build_model(self) -> None:
        self.model = Crossformer(
            data_dim=self.datamodule.num_features,
            in_len=self.task_config.windows,
            out_len=self.task_config.pred_len,
            seg_len=self.model_config.seg_len,
            win_size=self.model_config.win_size,
            factor=self.model_config.factor,
            d_model=self.model_config.d_model,
            d_ff=self.model_config.d_ff,
            n_heads=self.model_config.n_heads,
            e_layers=self.model_config.e_layers,
            dropout=self.model_config.dropout,
            baseline=self.model_config.baseline,
            device=self.runtime.device,
        ).to(self.runtime.device)
