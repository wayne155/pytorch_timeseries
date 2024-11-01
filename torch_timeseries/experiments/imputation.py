# import codecs
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
from torch_timeseries.dataloader.maskts import MaskTS
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *

from torch_timeseries.utils.model_stats import count_parameters
from ..utils.early_stop import EarlyStopping
from ..utils.parse_type import parse_type
from ..utils.reproduce import reproducible
from ..core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from ..dataloader import SlidingWindowTS
from ..utils import asdict_exc


@dataclass
class ImputationSettings:
    windows: int = 336
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    mask_rate : float = 0.5

@dataclass
class ImputationExp(BaseRelevant, BaseIrrelevant, ImputationSettings):
    loss_func_type : str = 'mse'
    def config_wandb(
        self,
        project: str,
        name: str,
        mode: str='online',
    ):
        import wandb
        # TODO: add seeds config parameters
        def convert_dict(dictionary):
            converted_dict = {}
            for key, value in dictionary.items():
                converted_dict[f"config.{key}"] = value
            return converted_dict

        # check wether this experiment had runned and reported on wandb
        api = wandb.Api()
        config_filter = convert_dict(self.result_related_config)
        runs = api.runs(path=project, filters=config_filter)
        
        try:
            if runs[0].state == "finished" or runs[0].state == "running":
                print(f"{self.model_type} {self.dataset_type} w{self.windows} m{self.mask_rate}  Experiment already reported, quiting...")
                self.finished = True
                return 
        except:
            pass
        m = self.model_type
        run = wandb.init(
            mode=mode,
            project=project,
            name=name,
            tags=[m, self.dataset_type, f"window-{self.windows}", f"m-{self.mask_rate}"],
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self



    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
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
            }
        )
        self.metrics.to(self.device)

    def _init_loss_func(self):
        loss_func_map = {'mse': MSELoss, 'mae': L1Loss}
        self.loss_func = parse_type(loss_func_map[self.loss_func_type], globals=globals())()

    def _init_model(self):
        NotImplementedError("not implemented!!!")

    @property
    def result_related_configs(self):
        ident = asdict_exc(self, BaseIrrelevant)
        return ident

    def _init_data_loader(self):
        self.dataset: TimeSeriesDataset = parse_type(self.dataset_type, globals=globals())(
            root=self.data_path
        )
        self.scaler = parse_type(self.scaler_type, globals=globals())()
        self.dataloader = MaskTS(
            self.dataset,
            self.scaler,
            window=self.windows,
            scale_in_train=True,
            shuffle_train=True,
            freq=self.dataset.freq,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")

    def _run_identifier(self, seed) -> str:
        ident = self.result_related_configs
        ident["seed"] = seed
        # only influence the evluation result, not included here
        # ident['invtrans_loss'] = False
        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return str(ident_md5)

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
            f"w{self.windows}m{self.mask_rate}",
            self._run_identifier(seed),
        )

        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )

        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )

        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )

    def _process_one_batch(
        self,
        batch_masked_x,
        batch_x,
        batch_origin_x,
        batch_mask,
        batch_x_date_enc,
    ):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # for single step you should output (B, N)
        # for multiple steps you should output (B, O, N)
        raise NotImplementedError()

    def _evaluate(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for (
                    batch_masked_x,
                    batch_x,
                    batch_origin_x,
                    batch_mask,
                    batch_x_date_enc,
                ) in dataloader:
                    batch_size = batch_x.size(0)
                    preds, truths = self._process_one_batch(
                        batch_masked_x, batch_x, batch_origin_x, batch_mask, batch_x_date_enc
                    )
                    
                    batch_mask = batch_mask.to(self.device)
                    batch_origin_x = batch_origin_x.to(self.device)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_x
                    if self.windows == 1:
                        self.metrics.update(
                            preds[batch_mask == 0].contiguous().reshape(batch_size, -1),
                            truths[batch_mask == 0].contiguous().reshape(batch_size, -1),
                        )
                    else:
                        self.metrics.update(preds[batch_mask == 0].contiguous(), truths[batch_mask == 0].contiguous())

                    progress_bar.update(batch_size)

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)

        if self._use_wandb():
            result = {}
            for name, metric_value in test_result.items():
                wandb.run.summary["test_" + name] = metric_value
                result["test_" + name] = metric_value
            wandb.log(result, step=self.current_epoch)

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Validating .... ")
        val_result = self._evaluate(self.val_loader)

        # log to wandb
        if self._use_wandb():
            result = {}
            for name, metric_value in val_result.items():
                wandb.run.summary["val_" + name] = metric_value
                result["val_" + name] = metric_value
            wandb.log(result, step=self.current_epoch)

        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                batch_masked_x,
                batch_x,
                batch_origin_x,
                batch_mask,
                batch_x_date_enc,
            ) in enumerate(self.train_loader):
                start = time.time()
                batch_origin_x = batch_origin_x.to(self.device)
                self.model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    batch_masked_x, batch_x, batch_origin_x, batch_mask , batch_x_date_enc
                )
                batch_mask = batch_mask.to(self.device)
                
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = batch_origin_x
                    
                loss = self.loss_func(pred[batch_mask == 0], true[batch_mask == 0])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(batch_x.size(0))
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
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
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

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _use_wandb(self):
        return hasattr(self, "wandb")

    def run(self, seed=42) -> Dict[str, float]:
        if hasattr(self, "finished") and self.finished is True:
            self._run_print("Experiment finished!!!")
            return {}

        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num

        # for resumable reproducibility
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")
            if self._use_wandb():
                wandb.log({'training_loss':np.mean(train_losses)}, step=self.current_epoch)

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)

            self._save_run_check_point(seed)

            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        return best_test_result

    def runs(self, seeds: List[int] = [1, 2, 3, 4, 5]):
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return
        # if self._use_wandb():
        #     wandb.config.update({"seeds": seeds})

        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            result = self.run(seed=seed)
            results.append(result)

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"
                wandb.finish()
        return self.metric_mean_std

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
