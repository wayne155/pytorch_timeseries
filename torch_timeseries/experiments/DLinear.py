from dataclasses import dataclass
import sys

import torch
from ..model import DLinear
from . import (
    ForecastExp,
    ImputationExp,
    UEAClassificationExp,
    AnomalyDetectionExp,
)


@dataclass
class DLinearParameters:
    individual: bool = False


@dataclass
class DLinearForecast(ForecastExp, DLinearParameters):
    model_type: str = "DLinear"

    def _run_engine_compat(self, seed):
        from torch_timeseries.experiment import Experiment

        kwargs = {
            "horizon": self.horizon,
            "windows": self.windows,
            "pred_len": self.pred_len,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "time_enc": self.time_enc,
            "input_columns": self.input_columns or None,
            "target_columns": self.target_columns or None,
            "individual": self.individual,
            "data_path": self.data_path,
            "save_dir": self.save_dir,
            "device": self.device,
            "num_worker": self.num_worker,
            "scaler_type": self.scaler_type,
            "optm_type": self.optm_type,
            "loss_func_type": self.loss_func_type,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "l2_weight_decay": self.l2_weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "max_grad_norm": self.max_grad_norm,
            "invtrans_loss": self.invtrans_loss,
        }
        result = Experiment(
            model="DLinear",
            task="Forecast",
            dataset=self.dataset_type,
            **kwargs,
        ).run(seeds=[seed])
        return result[0].metrics

    def run(self, seed=42):
        return self._run_engine_compat(seed)

    def runs(self, seeds=None):
        seeds = [1, 2, 3, 4, 5] if seeds is None else seeds
        return [self.run(seed=seed) for seed in seeds]


@dataclass
class DLinearUEAClassification(UEAClassificationExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
            output_prob=self.dataset.num_classes
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y.long().squeeze(-1)


@dataclass
class DLinearAnomalyDetection(AnomalyDetectionExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, origin_x, batch_y):
        # inputs:
            # batch_x: (B, T, N)
            # origin_x: (B, T, N)
        # ouputs:
        # - pred: (B, O, N)
        # - label: (B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x


@dataclass
class DLinearImputation(ImputationExp, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.windows,
            enc_in=self.dataset.num_features,
            individual=self.individual,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(
        self,
        batch_masked_x,
        batch_x,
        batch_origin_x,
        batch_mask,
        batch_x_date_enc,
    ):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_masked_x = batch_masked_x.to(self.device, dtype=torch.float32)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        outputs = self.model(
            batch_masked_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_x

