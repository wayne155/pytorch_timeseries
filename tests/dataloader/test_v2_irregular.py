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
    with pytest.raises(AssertionError, match="x_time"):
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
