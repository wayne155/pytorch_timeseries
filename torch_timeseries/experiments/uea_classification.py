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
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, Accuracy
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import *
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *

from torch_timeseries.utils.acc import accuracy
from torch_timeseries.utils.model_stats import count_parameters
from ..utils.early_stop import EarlyStopping
from ..utils.parse_type import parse_type
from ..utils.reproduce import reproducible
from ..core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from ..dataloader import UEAClassification
from ..utils import asdict_exc


@dataclass
class ClassificationSettings:
    windows: int = 336

@dataclass
class UEAClassificationExp(BaseRelevant, BaseIrrelevant, ClassificationSettings):
    
    loss_func_type : str = 'cross_entropy'
    
    def config_wandb(
        self,
        project: str,
        name: str,
        mode: str = "online",
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
                print(
                    f"{self.model_type} {self.dataset_type} w{self.windows} Experiment already reported, quiting..."
                )
                self.finished = True
                return
        except:
            pass
        m = self.model_type
        run = wandb.init(
            mode=mode,
            project=project,
            name=name,
            tags=[m, self.dataset_type, f"window-{self.windows}"],
        )
        wandb.config.update(asdict(self))
        self.wandb = True
        print(f"using wandb , running in config: {asdict(self)}")
        return self

    def _init_optimizer(self):
        self.optimizer = parse_type(self.optm_type, globals=globals())(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
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
                "accuracy": Accuracy("multiclass", num_classes=self.dataset.num_classes),
            }
        )
        self.metrics.to(self.device)

    def _init_loss_func(self):
        loss_func_map = {"cross_entropy": CrossEntropyLoss}
        self.loss_func = parse_type(
            loss_func_map[self.loss_func_type], globals=globals()
        )()

    def _init_model(self):
        NotImplementedError("not implemented!!!")

    @property
    def result_related_configs(self):
        ident = asdict_exc(self, BaseIrrelevant)
        return ident

    def _init_data_loader(self):
        self.dataset: UEA = UEA(self.dataset_type, root=self.data_path)
        self.scaler = parse_type(self.scaler_type, globals=globals())()
        self.dataloader = UEAClassification(
            self.dataset,
            self.scaler,
            window=self.windows,
            shuffle_train=True,
            batch_size=self.batch_size,
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )

        print(f"train steps: {len(self.train_loader.dataset)}")
        print(f"val steps: {len(self.val_loader.dataset)}")
        print(f"test steps: {len(self.test_loader.dataset)}")

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
            f"w{self.windows}",
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
        batch_x,
        batch_y,
        batch_origin_x,
        batch_origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
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

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()

        losses = []

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for scaled_x, x, y, padding_masks in dataloader:
                    batch_size = scaled_x.size(0)
                    unormed_probs, truths = self._process_one_batch(
                        scaled_x, x, y, padding_masks
                    )
                    loss = self.loss_func(unormed_probs, truths.long().squeeze(-1))
                    losses.append(loss.item())
                    # probs = torch.nn.functional.softmax(
                    #     unormed_probs
                    # )  # (total_samples, num_classes) est. prob. for each class and sample
                    # predictions = torch.argmax(probs, dim=1)
                    # print("acc", accuracy(unormed_probs, y))
                    self.metrics.update(unormed_probs, truths)
                    progress_bar.update(batch_size)

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
            result[self.loss_func_type] = np.mean(losses)
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
            for i, (scaled_x, x, y, padding_masks) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                pred, true = self._process_one_batch(
                    scaled_x, x, y, padding_masks
                )
                loss = self.loss_func(pred, true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(scaled_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.optimizer.step()

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
        self.optimizer.load_state_dict(check_point["optimizer"])
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
            "optimizer": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
