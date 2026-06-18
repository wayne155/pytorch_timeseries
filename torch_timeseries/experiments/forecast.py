# import codecs
from dataclasses import asdict, dataclass, field
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *

from torch_timeseries.utils.model_stats import count_parameters
from ..utils.early_stop import EarlyStopping
from ..utils.parse_type import parse_type
from ..utils.reproduce import get_rng_state, reproducible, set_rng_state
from ..core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from ..dataloader.v2 import ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig, TimeEncConfig, TSBatch
from ..utils import asdict_exc

try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: wandb is not installed, some functionality may not work.")
@dataclass
class ForecastSettings:
    horizon: int = 1
    windows: int = 96
    pred_len: int = 96
    # Both None -> the dataset's default split (canonical borders for the
    # ETT family, 7:1:2 ratios otherwise). Set explicitly to override.
    train_ratio: Optional[float] = None
    test_ratio: Optional[float] = None
    time_enc: int = 1
    input_columns: List[int] = field(default_factory=list)
    target_columns: List[int] = field(default_factory=list)
    
@dataclass
class ForecastExp(BaseRelevant, BaseIrrelevant, ForecastSettings):
    loss_func_type : str = 'mse'
    columns : List[int] = field(default_factory=lambda : [])
    # TSLib protocol: 10 epochs, patience 3, halve lr each epoch, no clipping.
    epochs: int = 10
    patience: int = 3
    lradj: str = 'type1'
    max_grad_norm: Optional[float] = None
    
    def config_wandb(
        self,
        project: str,
    ):
        self.project = project
        self.wandb = True
        return self

    def _init_wandb(
        self,
        project: str,
        seed : int
    ):
        # TODO: add seeds config parameters
        def convert_dict(dictionary):
            converted_dict = {}
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict
        # check wether this experiment had runned and reported on wandb
        api = wandb.Api()
        dic = dict(self.result_related_configs)
        dic.update({'seed': seed})
        config_filter = convert_dict(dic)
        runs = api.runs(path=project, filters=config_filter)
        try:
            for run in runs:
                if run.state == "finished" or run.state == "running":
                    print(f"seed-{seed} {self.model_type} {self.dataset_type} w{self.windows} p{self.pred_len}  Experiment already reported, quiting...")
                    self.finished = True
                    return False 
        except:
            pass
        wandb.init(
            mode="online",
            project=project,
            name=f"{self.model_type}-{self.dataset_type}-w{self.windows} p{self.pred_len}",
            tags=[self.dataset_type, f"horizon-{self.horizon}", f"window-{self.windows}", f"pred-{self.pred_len}"],
        )
        wandb.config.update(asdict(self))
        wandb.config.update({'seed': seed})
        print(f" wandb config , running in config: {asdict(self)}")
        return True
    

    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        if self.lradj == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optim, T_max=self.epochs
            )
        else:
            # TSLib "type1": halve the lr every epoch (first adjust is a no-op,
            # so epoch 0 and 1 run at the base lr).
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.model_optim,
                lr_lambda=lambda e: 0.5 ** max(0, e - 1),
            )

    def _setup(self):
        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        # init loss function based on given loss func type
        self._init_loss_func()

        self.current_epochs = 0
        self.current_run = 0

        self.setuped = True

    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rmse": MeanSquaredError(squared=False),
            }
        )
        self.metrics.to(self.device)

    @property
    def monitor_metric(self) -> str:
        """Validation metric used for early stopping / model selection."""
        return self.loss_func_type

    def _init_loss_func(self):
        loss_func_map = {'mse': MSELoss, 'mae': L1Loss}
        self.loss_func = parse_type(loss_func_map[self.loss_func_type], globals=globals())()

    def _init_model(self):
        NotImplementedError("not implemented!!!")

    @property
    def result_related_configs(self):
        ident = asdict_exc(self, BaseIrrelevant)
        return ident

    def _init_dataset(self):
        self.dataset: TimeSeriesDataset = parse_type(self.dataset_type, globals())(
            root=self.data_path, columns=self.columns
        )

    
    def _init_data_loader(self):
        self._init_dataset()
        self.scaler = parse_type(self.scaler_type, globals=globals())()

        window_cfg = WindowConfig(
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            time_enc_cfg=TimeEncConfig(
                time_enc=self.time_enc,
                freq=getattr(self.dataset, "freq", None),
            ),
            input_columns=self.input_columns or None,
            target_columns=self.target_columns or None,
        )
        # None -> the dataset's default split (canonical borders or 7:1:2)
        if self.train_ratio is None and self.test_ratio is None:
            split_cfg = None
        else:
            split_cfg = SplitConfig(
                train=self.train_ratio if self.train_ratio is not None else 0.7,
                test=self.test_ratio,
            )
        loader_cfg = LoaderConfig(
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle_train=True,
        )
        self.datamodule = ForecastDataModule(
            dataset=self.dataset,
            scaler=self.scaler,
            window=window_cfg,
            split=split_cfg,
            loader=loader_cfg,
        )
        self.train_loader = self.datamodule.train_loader
        self.val_loader   = self.datamodule.val_loader
        self.test_loader  = self.datamodule.test_loader
        self.train_steps  = len(self.train_loader.dataset)
        self.val_steps    = len(self.val_loader.dataset)
        self.test_steps   = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps:   {self.val_steps}")
        print(f"test steps:  {self.test_steps}")

    def _run_identifier(self, seed) -> str:
        ident = self.result_related_configs
        # ident["seed"] = seed
        # only influence the evluation result, not included here
        # ident['invtrans_loss'] = False
        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{seed}-{str(ident_md5)}"

    def _setup_run(self, seed):
        # setup experiment  only once
        if not hasattr(self, "setuped"):
            self._setup()
        # setup torch and numpy random seed
        reproducible(seed)
        # init model, optimizer and loss function
        self._init_model()

        self._init_optimizer()

        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
            self._run_identifier(seed),
        )
        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )
        self._setup_early_stopper()
        
        
    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )
        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )

    def _process_one_batch(self, batch: TSBatch):
        """Run the model on one TSBatch.

        Useful batch fields (see ``torch_timeseries.dataloader.v2.TSBatch``):
            batch.x / batch.y                      scaled input / target, (B, T, N) / (B, O, N)
            batch.x_raw / batch.y_raw              unscaled values
            batch.x_time_feature / y_time_feature  encoded time features

        Returns:
            (pred, true): (B, O, N) each — or (B, N) for single-step.
        """
        batch_x = batch.x.to(self.device, dtype=torch.float32)
        batch_y = batch.y.to(self.device, dtype=torch.float32)
        return self.model(batch_x), batch_y

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for batch in dataloader:
                    batch_size = batch.x.size(0)
                    preds, truths = self._process_one_batch(batch)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch.y_raw.to(self.device)
                    if self.pred_len == 1:
                        self.metrics.update(
                            preds.contiguous().reshape(batch_size, -1),
                            truths.contiguous().reshape(batch_size, -1),
                        )
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())

                    progress_bar.update(batch_size)

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)

        # if self._use_wandb():
        #     import wandb
        #     result = {}
        #     for name, metric_value in test_result.items():
        #         wandb.run.summary["test_" + name] = metric_value
        #         result["test_" + name] = metric_value
        #     wandb.log(result, step=self.current_epoch)

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Validating .... ")
        val_result = self._evaluate(self.val_loader)

        # # log to wandb
        # if self._use_wandb():
        #     import wandb
        #     result = {}
        #     for name, metric_value in val_result.items():
        #         wandb.run.summary["val_" + name] = metric_value
        #         result["val_" + name] = metric_value
        #     wandb.log(result, step=self.current_epoch)

        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, batch in enumerate(self.train_loader):
                self.model_optim.zero_grad()
                pred, true = self._process_one_batch(batch)
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = batch.y_raw.to(self.device)
                loss = self.loss_func(pred, true)
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

    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with open(
            os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device, weights_only=False)
        )

    def _run_print(self, *args, **kwargs):
        time = (
            "["
            + str(datetime.datetime.now() + datetime.timedelta(hours=8))[:19]
            + "] -"
        )
        print(*args, **kwargs)
        with open(os.path.join(self.run_save_dir, "output.log"), "a+") as f:
            print(time, *args, flush=True, file=f)

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        if "rng_state" in check_point:
            rng_state = check_point["rng_state"]
            if isinstance(rng_state, dict):
                set_rng_state(rng_state)
            elif isinstance(rng_state, torch.Tensor):
                torch.set_rng_state(rng_state.cpu())
            else:
                torch.set_rng_state(rng_state)

        self.early_stopper.set_state(check_point["early_stopping"])

    def _use_wandb(self):
        return hasattr(self, "wandb")

    def run(self, seed=42) -> Dict[str, float]:

        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num

        # for resumable reproducibility
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.monitor_metric], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result

    def runs(self, seeds: List[int] = [1, 2, 3, 4, 5]):
        results = []
        for i, seed in enumerate(seeds):
            result = self.run(seed=seed)
            results.append(result)

        return results

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)


        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
