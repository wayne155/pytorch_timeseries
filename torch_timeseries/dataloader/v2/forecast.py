"""Task-level convenience layer.

Compared to v1's ``SlidingWindowTS``, this module:

* Accepts a single ``WindowConfig`` and ``SplitConfig`` instead of ~20 kwargs.
* Returns ``TSBatch`` from each loader, so adding a new modality (e.g. exogenous
  variables) does not break unpacking sites.
* The same class handles overlapping and non-overlapping windowing via
  ``WindowConfig.stride``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import DataLoader

from torch_timeseries.core import TimeSeriesDataset, TimeseriesSubset

from .._seed import seed_worker
from .._split import resolve_split_ratios
from .batch import TimeEncConfig, collate_tsbatch
from .windowed import WindowedDataset


@dataclass
class WindowConfig:
    window: int = 96
    horizon: int = 1
    steps: int = 96
    stride: int = 1
    fast_val: bool = False
    fast_test: bool = False
    time_enc_cfg: TimeEncConfig = field(default_factory=TimeEncConfig)
    input_columns: Optional[list] = None
    target_columns: Optional[list] = None


@dataclass
class SplitConfig:
    train: float = 0.7
    val: Optional[float] = None
    test: Optional[float] = 0.2
    uniform_eval: bool = True
    """If True, val/test subsets are extended by ``window+horizon-1`` so that
    every sample in the original split has a full lookback window."""


@dataclass
class LoaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: bool = False


class ForecastDataModule:
    """Wires dataset -> scaler -> split -> WindowedDataset -> DataLoader."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler,
        window: WindowConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
        scale_in_train: bool = True,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or WindowConfig()
        self.split_cfg = split or SplitConfig()
        self.loader_cfg = loader or LoaderConfig()
        self.scale_in_train = scale_in_train

        self._build_subsets()
        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    # ------------------------------------------------------------------ #
    # construction steps                                                 #
    # ------------------------------------------------------------------ #

    def _build_subsets(self) -> None:
        train, val, test = resolve_split_ratios(
            train_ratio=self.split_cfg.train,
            val_ratio=self.split_cfg.val,
            test_ratio=self.split_cfg.test,
        )
        self.train_ratio, self.val_ratio, self.test_ratio = train, val, test

        n = len(self.dataset)
        train_size = int(train * n)
        test_size = int(test * n)
        val_size = n - train_size - test_size

        wc = self.window_cfg
        eval_pad = wc.window + wc.horizon - 1 if self.split_cfg.uniform_eval else 0

        idx = range(n)
        self.train_subset = TimeseriesSubset(self.dataset, idx[0:train_size])
        self.val_subset = TimeseriesSubset(
            self.dataset,
            idx[train_size - eval_pad: train_size + val_size],
        )
        self.test_subset = TimeseriesSubset(
            self.dataset, idx[-(test_size + eval_pad):] if eval_pad else idx[-test_size:]
        )

    def _fit_scaler(self) -> None:
        if self.scale_in_train:
            self.scaler.fit(self.train_subset.data)
        else:
            self.scaler.fit(self.dataset.data)

    def _build_datasets(self) -> None:
        wc = self.window_cfg
        non_overlap_stride = wc.window + wc.horizon + wc.steps - 1
        common = dict(
            scaler=self.scaler,
            window=wc.window,
            horizon=wc.horizon,
            steps=wc.steps,
            time_enc_cfg=wc.time_enc_cfg,
            input_columns=wc.input_columns,
            target_columns=wc.target_columns,
        )
        self.train_dataset = WindowedDataset(self.train_subset, stride=wc.stride, **common)
        self.val_dataset = WindowedDataset(
            self.val_subset,
            stride=non_overlap_stride if wc.fast_val else wc.stride,
            **common,
        )
        self.test_dataset = WindowedDataset(
            self.test_subset,
            stride=non_overlap_stride if wc.fast_test else wc.stride,
            **common,
        )

    def _build_loaders(self) -> None:
        lc = self.loader_cfg
        kw = dict(
            batch_size=lc.batch_size,
            num_workers=lc.num_workers,
            pin_memory=lc.pin_memory,
            collate_fn=collate_tsbatch,
            worker_init_fn=seed_worker,
        )
        self.train_loader = DataLoader(self.train_dataset, shuffle=lc.shuffle_train, **kw)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **kw)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **kw)

    # ------------------------------------------------------------------ #
    # passthrough conveniences                                           #
    # ------------------------------------------------------------------ #

    @property
    def window(self) -> int:
        return self.window_cfg.window

    @property
    def horizon(self) -> int:
        return self.window_cfg.horizon

    @property
    def steps(self) -> int:
        return self.window_cfg.steps

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_target_features(self) -> int:
        wc = self.window_cfg
        if wc.target_columns is not None:
            return len(wc.target_columns)
        return self.dataset.num_features
