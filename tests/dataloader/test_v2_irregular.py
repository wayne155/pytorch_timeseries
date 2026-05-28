import torch


def test_irregular_tsbatch_collation():
    """Variable-length batch pads correctly; mask shape correct; t=1.0 at padded positions."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F = 3
    samples = []
    for T in [4, 6, 5]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = (torch.rand(T, F) > 0.3).float()
        y = torch.tensor(0, dtype=torch.long)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, y=y))

    batch = collate_irregular(samples)

    assert batch.x.shape == (3, 6, F)
    assert batch.t.shape == (3, 6)
    assert batch.mask.shape == (3, 6, F)
    assert batch.y.shape == (3,)
    # sample 0 (T=4): positions 4 and 5 must be padded → mask=0
    assert batch.mask[0, 4:, :].sum() == 0
    # sample 1 (T=6): no padding → has real observations
    assert batch.mask[1, :, :].sum() > 0
    # padded t positions must be 1.0
    assert (batch.t[0, 4:] == 1.0).all()


def test_collate_irregular_with_query_times():
    """Query-time fields (for interp/forecast) are also padded correctly."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F = 2
    samples = []
    for T, Tq in [(4, 2), (6, 3), (5, 2)]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = torch.ones(T, F)
        y = torch.randn(Tq, F)
        t_query = torch.linspace(0.5, 1.0, Tq)
        query_mask = torch.ones(Tq, F)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, y=y,
                                         t_query=t_query, query_mask=query_mask))

    batch = collate_irregular(samples)
    assert batch.t_query.shape == (3, 3)
    assert batch.query_mask.shape == (3, 3, F)
    # sample 0 and 2 (Tq=2): position 2 padded → query_mask=0
    assert batch.query_mask[0, 2, :].sum() == 0
    assert batch.query_mask[2, 2, :].sum() == 0


def test_collate_irregular_1d_y():
    """1-D y (e.g. multi-label vector) is stacked correctly without IndexError."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F, C = 3, 4
    samples = []
    for T in [4, 6, 5]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = torch.ones(T, F)
        y = torch.randn(C)   # 1-D, shape (C,)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, y=y))
    batch = collate_irregular(samples)
    assert batch.y.shape == (3, C)


def test_collate_irregular_x_time():
    """x_time calendar features are padded correctly and produce (B, max_T, C) batch."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F, C = 3, 4
    samples = []
    for T in [4, 6, 5]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = torch.ones(T, F)
        x_time = torch.randn(T, C)
        samples.append(IrregularTSBatch(x=x, t=t, mask=mask, x_time=x_time))
    batch = collate_irregular(samples)
    assert batch.x_time.shape == (3, 6, C)
    # padded positions (sample 0: positions 4-5) must be zero
    assert (batch.x_time[0, 4:, :] == 0).all()


def test_collate_irregular_partial_x_time_raises():
    """collate_irregular raises AssertionError when only some samples have x_time."""
    import pytest
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F, C = 3, 4
    T = 5
    s_with = IrregularTSBatch(
        x=torch.randn(T, F), t=torch.linspace(0.0, 1.0, T),
        mask=torch.ones(T, F), x_time=torch.randn(T, C),
    )
    s_without = IrregularTSBatch(
        x=torch.randn(T, F), t=torch.linspace(0.0, 1.0, T),
        mask=torch.ones(T, F), x_time=None,
    )
    with pytest.raises(AssertionError, match="All samples must have x_time"):
        collate_irregular([s_with, s_without])


def test_collate_irregular_t_query_time():
    """t_query_time calendar features at query times are padded correctly."""
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch, collate_irregular
    F, C = 2, 3
    samples = []
    for T, Tq in [(4, 2), (6, 4), (5, 3)]:
        x = torch.randn(T, F)
        t = torch.linspace(0.0, 1.0, T)
        mask = torch.ones(T, F)
        t_query = torch.linspace(0.5, 1.0, Tq)
        query_mask = torch.ones(Tq, F)
        t_query_time = torch.randn(Tq, C)
        y = torch.randn(Tq, F)
        samples.append(IrregularTSBatch(
            x=x, t=t, mask=mask, y=y,
            t_query=t_query, query_mask=query_mask, t_query_time=t_query_time,
        ))
    batch = collate_irregular(samples)
    assert batch.t_query_time.shape == (3, 4, C)
    # sample 0 (Tq=2): positions 2-3 padded → zero
    assert (batch.t_query_time[0, 2:, :] == 0).all()


import numpy as np


class _ToyIrregular:
    """In-memory irregular dataset — no file I/O."""
    num_features = 3
    num_classes = 2

    def __init__(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        self.samples, self.times, self.masks, self.labels = [], [], [], []
        for i in range(n):
            T = rng.integers(5, 15)
            x = rng.normal(size=(T, self.num_features)).astype(np.float32)
            t = np.sort(rng.uniform(0, 48, size=T)).astype(np.float32)
            mask = (rng.random((T, self.num_features)) > 0.3).astype(np.float32)
            self.samples.append(x)
            self.times.append(t)
            self.masks.append(mask)
            self.labels.append(i % self.num_classes)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)


def _toy_irreg_cls_dm(**kw):
    from torch_timeseries.dataloader.v2.irregular_classification import (
        IrregularClassificationDataModule, IrregularClassificationConfig,
    )
    from torch_timeseries.dataloader.v2.forecast import SplitConfig, LoaderConfig
    from torch_timeseries.scaler import StandardScaler
    return IrregularClassificationDataModule(
        dataset=_ToyIrregular(n=40),
        scaler=StandardScaler(),
        window=IrregularClassificationConfig(**kw),
        split=SplitConfig(train=0.7, val=0.1, test=0.2),
        loader=LoaderConfig(batch_size=8, num_workers=0),
    )


def test_irregular_classification_dm_returns_batch():
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    from torch_timeseries.dataloader.v2.irregular_batch import IrregularTSBatch
    assert isinstance(batch, IrregularTSBatch)


def test_irregular_classification_dm_batch_shapes():
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    B = batch.x.shape[0]
    T = batch.x.shape[1]
    assert batch.x.shape == (B, T, 3)
    assert batch.t.shape == (B, T)
    assert batch.mask.shape == (B, T, 3)
    assert batch.y.shape == (B,)
    assert batch.y.dtype == torch.long


def test_irregular_classification_dm_t_normalized():
    """All t values should be in [0, 1] (padded positions use 1.0 sentinel)."""
    dm = _toy_irreg_cls_dm()
    batch = next(iter(dm.train_loader))
    assert batch.t.min() >= 0.0
    assert batch.t.max() <= 1.0


def test_irregular_classification_dm_properties():
    dm = _toy_irreg_cls_dm()
    assert dm.num_features == 3
    assert dm.num_classes == 2
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert hasattr(dm, "test_dataset")
