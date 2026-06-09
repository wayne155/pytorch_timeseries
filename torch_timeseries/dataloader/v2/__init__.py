"""Composable, dict-batched dataloader API (preview).

Rough shape of the new API::

    from torch_timeseries.dataloader.v2 import (
        ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig,
    )

    dm = ForecastDataModule(
        dataset=ETTh1("./data"),
        scaler=StandardScaler(),
        window=WindowConfig(window=96, horizon=1, steps=336),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=32),
    )

    for batch in dm.train_loader:
        # batch is a TSBatch, never a positional tuple
        pred = model(batch.x.float())
        loss = mse(pred, batch.y.float())

The legacy ``SlidingWindowTS`` etc. are unchanged in the parent package.
"""
from .batch import TSBatch, Time, TimeEncConfig, collate_tsbatch
from .irregular_batch import IrregularTSBatch, collate_irregular
from .irregular_classification import IrregularClassificationDataModule, IrregularClassificationConfig
from .windowed import WindowedDataset
from .forecast import ForecastDataModule, WindowConfig, SplitConfig, LoaderConfig
from .imputation import ImputationDataModule, ImputationWindowConfig
from .anomaly import AnomalyDataModule, AnomalyWindowConfig
from .uea import UEADataModule, UEAWindowConfig
from torch_timeseries.utils.timefeatures import TimeEncoding

__all__ = [
    "TSBatch",
    "Time",
    "TimeEncConfig",
    "collate_tsbatch",
    "WindowedDataset",
    "ForecastDataModule",
    "WindowConfig",
    "SplitConfig",
    "LoaderConfig",
    "TimeEncoding",
    "ImputationDataModule",
    "ImputationWindowConfig",
    "AnomalyDataModule",
    "AnomalyWindowConfig",
    "UEADataModule",
    "UEAWindowConfig",
    "IrregularTSBatch",
    "collate_irregular",
    "IrregularClassificationDataModule",
    "IrregularClassificationConfig",
]
