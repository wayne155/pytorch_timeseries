"""Task-level convenience layer.

Compared to v1's ``SlidingWindowTS``, this module:

* Accepts a single ``WindowConfig`` and ``SplitConfig`` instead of ~20 kwargs.
* Returns ``TSBatch`` from each loader, so adding a new modality (e.g. exogenous
  variables) does not break unpacking sites.
* The same class handles overlapping and non-overlapping windowing via
  ``WindowConfig.stride``.
"""
from __future__ import annotations

import warnings

from torch.utils.data import DataLoader

from torch_timeseries.core import TimeSeriesDataset, TimeseriesSubset

from .._seed import seed_worker
from .._split import resolve_split_ratios
from .batch import collate_tsbatch
from .loader import LoaderConfig
from .split import SplitConfig, default_split_config
from .window import WindowConfig
from .windowed import WindowedDataset


class ForecastDataModule:
    """Wires dataset -> scaler -> split -> WindowedDataset -> DataLoader."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        scaler,
        window: WindowConfig = None,
        split: SplitConfig = None,
        loader: LoaderConfig = None,
    ) -> None:
        self.dataset = dataset
        self.scaler = scaler
        self.window_cfg = window or WindowConfig()
        # No split config -> the dataset's default (canonical borders if known)
        self._split_from_default = split is None
        self.split_cfg = split if split is not None else default_split_config(dataset)
        self.loader_cfg = loader or LoaderConfig()

        self._build_subsets()
        self._fit_scaler()
        self._build_datasets()
        self._build_loaders()

    # ------------------------------------------------------------------ #
    # construction steps                                                 #
    # ------------------------------------------------------------------ #

    def _build_subsets(self) -> None:
        wc = self.window_cfg
        sc = self.split_cfg
        eval_pad = wc.window + wc.horizon - 1 if sc.uniform_eval else 0
        n = len(self.dataset)
        idx = range(n)

        # Precedence: explicit borders > dataset's canonical borders > ratios.
        borders = sc.borders
        borders_from_default = self._split_from_default
        if borders is None and sc.use_dataset_borders:
            borders = default_split_config(self.dataset).borders
            borders_from_default = True
        if borders is not None:
            train_end, val_end, test_end = borders
            if not (0 < train_end <= val_end <= test_end <= n):
                if not borders_from_default:
                    raise ValueError(
                        f"invalid split borders {borders} for dataset of length {n}"
                    )
                # e.g. a truncated/mocked copy of a benchmark dataset — its
                # canonical borders no longer fit, use the ratio split instead.
                warnings.warn(
                    f"canonical split borders {borders} exceed dataset "
                    f"length {n}; falling back to the ratio split"
                )
                borders = None
        if borders is not None:
            train_end, val_end, test_end = borders
            self.train_ratio = train_end / n
            self.val_ratio = (val_end - train_end) / n
            self.test_ratio = (test_end - val_end) / n
            self.train_subset = TimeseriesSubset(self.dataset, idx[0:train_end])
            self.val_subset = TimeseriesSubset(
                self.dataset, idx[train_end - eval_pad: val_end]
            )
            self.test_subset = TimeseriesSubset(
                self.dataset, idx[val_end - eval_pad: test_end]
            )
            return

        train, val, test = resolve_split_ratios(
            train_ratio=self.split_cfg.train,
            val_ratio=self.split_cfg.val,
            test_ratio=self.split_cfg.test,
        )
        self.train_ratio, self.val_ratio, self.test_ratio = train, val, test

        train_size = int(train * n)
        test_size = int(test * n)
        val_size = n - train_size - test_size

        self.train_subset = TimeseriesSubset(self.dataset, idx[0:train_size])
        self.val_subset = TimeseriesSubset(
            self.dataset,
            idx[train_size - eval_pad: train_size + val_size],
        )
        self.test_subset = TimeseriesSubset(
            self.dataset, idx[-(test_size + eval_pad):] if eval_pad else idx[-test_size:]
        )

    def _fit_scaler(self) -> None:
        # Always fit on train only — fitting on val/test data leaks statistics.
        self.scaler.fit(self.train_subset.data)

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
