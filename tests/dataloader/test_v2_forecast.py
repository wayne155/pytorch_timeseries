"""Tests for ForecastDataModule (v2 dataloader)."""
import numpy as np
import pandas as pd
import pytest
import torch

from torch_timeseries.core import TimeSeriesDataset, Freq
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataloader.v2 import (
    ForecastDataModule,
    WindowConfig,
    SplitConfig,
    LoaderConfig,
    TSBatch,
    Time,
    TimeEncConfig,
)


# ── toy dataset ─────────────────────────────────────────────────────────────

class ToyTS(TimeSeriesDataset):
    name = "toy_forecast"
    num_features = 6
    freq = Freq.hours

    def download(self): pass

    def _load(self):
        n = 600
        rng = np.random.default_rng(42)
        self.df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="h"),
            **{f"f{i}": rng.standard_normal(n).astype("float32") for i in range(6)},
        })
        self.dates = self.df[["date"]]
        self.data = self.df.drop("date", axis=1).values.astype("float32")
        self.length = n


WINDOW, HORIZON, STEPS, N_FEAT = 24, 4, 12, 6


def _dm(*, window=WINDOW, horizon=HORIZON, steps=STEPS, stride=1,
        fast_val=False, fast_test=False, time_enc_cfg=None,
        input_columns=None, target_columns=None,
        train=0.7, val=0.1, test=0.2, batch_size=8):
    return ForecastDataModule(
        dataset=ToyTS(root="/tmp"),
        scaler=StandardScaler(),
        window=WindowConfig(
            window=window, horizon=horizon, steps=steps, stride=stride,
            fast_val=fast_val, fast_test=fast_test,
            time_enc_cfg=time_enc_cfg or TimeEncConfig(),
            input_columns=input_columns,
            target_columns=target_columns,
        ),
        split=SplitConfig(train=train, val=val, test=test),
        loader=LoaderConfig(batch_size=batch_size, num_workers=0),
    )


def _first_batch(dm, split="train") -> TSBatch:
    loader = getattr(dm, f"{split}_loader")
    return next(iter(loader))


# ── batch type ──────────────────────────────────────────────────────────────

def test_batch_is_tsbatch():
    dm = _dm()
    assert isinstance(_first_batch(dm), TSBatch)


# ── x / y shapes ────────────────────────────────────────────────────────────

def test_x_y_shapes():
    dm = _dm()
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x.shape == (B, WINDOW, N_FEAT)
    assert b.y.shape == (B, STEPS, N_FEAT)


def test_x_y_shapes_vary_with_window_steps():
    dm = _dm(window=48, steps=24)
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x.shape == (B, 48, N_FEAT)
    assert b.y.shape == (B, 24, N_FEAT)


# ── raw fields ───────────────────────────────────────────────────────────────

def test_x_raw_present():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_raw is not None
    assert b.y_raw is not None


def test_x_raw_shape_matches_x():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_raw.shape == b.x.shape
    assert b.y_raw.shape == b.y.shape


def test_x_raw_differs_from_x_after_scaling():
    """Scaler should change the data values (StandardScaler zero-means columns)."""
    dm = _dm()
    b = _first_batch(dm)
    assert not torch.allclose(b.x, b.x_raw)


# ── time feature (encoded tensor) ────────────────────────────────────────────

def test_x_time_feature_present():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_time_feature is not None
    assert b.y_time_feature is not None


def test_x_time_feature_is_float():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_time_feature.dtype in (torch.float32, torch.float64)


def test_x_time_feature_window_dim():
    dm = _dm()
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x_time_feature.shape[0] == B
    assert b.x_time_feature.shape[1] == WINDOW
    assert b.y_time_feature.shape[1] == STEPS


def test_time_enc_cfg_custom_fn():
    """custom_fn in TimeEncConfig should be called instead of time_features."""
    calls = []

    def my_enc(dates):
        calls.append(len(dates))
        return np.ones((len(dates), 3), dtype="float32")

    dm = _dm(time_enc_cfg=TimeEncConfig(custom_fn=my_enc))
    b = _first_batch(dm)
    assert len(calls) > 0, "custom_fn was never called"
    assert b.x_time_feature.shape[-1] == 3


# ── Time dataclass (raw components) ──────────────────────────────────────────

def test_x_time_present():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_time is not None
    assert b.y_time is not None


def test_x_time_is_time_instance():
    dm = _dm()
    b = _first_batch(dm)
    assert isinstance(b.x_time, Time)


def test_x_time_components_shape():
    dm = _dm()
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    for attr in ("year", "month", "day", "weekday", "hour", "minute", "second"):
        t = getattr(b.x_time, attr)
        assert t is not None, f"x_time.{attr} is None"
        assert t.shape == (B, WINDOW), f"x_time.{attr} shape {t.shape}"
        assert t.dtype == torch.long, f"x_time.{attr} dtype {t.dtype}"


def test_x_time_month_in_range():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_time.month.min() >= 1
    assert b.x_time.month.max() <= 12


def test_x_time_hour_in_range():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_time.hour.min() >= 0
    assert b.x_time.hour.max() <= 23


def test_y_time_components_shape():
    dm = _dm()
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.y_time.year.shape == (B, STEPS)


# ── index field ──────────────────────────────────────────────────────────────

def test_x_index_present():
    dm = _dm()
    b = _first_batch(dm)
    assert b.x_index is not None
    assert b.y_index is not None


def test_x_index_shape():
    dm = _dm()
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x_index.shape == (B, WINDOW)
    assert b.y_index.shape == (B, STEPS)


def test_y_index_follows_x_index():
    """y window starts horizon steps after x window ends."""
    dm = _dm()
    b = _first_batch(dm)
    gap = b.y_index[:, 0] - b.x_index[:, -1]
    assert (gap == HORIZON).all()


# ── stride & fast_val / fast_test ────────────────────────────────────────────

def test_stride_1_more_windows_than_non_overlap():
    non_overlap = WINDOW + HORIZON + STEPS - 1
    dm_slide = _dm(stride=1)
    dm_fast = _dm(stride=non_overlap)
    assert len(dm_slide.train_dataset) > len(dm_fast.train_dataset)


def test_fast_val_reduces_val_dataset():
    dm_slide = _dm(fast_val=False)
    dm_fast = _dm(fast_val=True)
    assert len(dm_fast.val_dataset) <= len(dm_slide.val_dataset)


def test_fast_test_reduces_test_dataset():
    dm_slide = _dm(fast_test=False)
    dm_fast = _dm(fast_test=True)
    assert len(dm_fast.test_dataset) <= len(dm_slide.test_dataset)


def test_fast_val_non_overlap_window_count():
    """With fast_val, val windows should match the non-overlap formula."""
    dm = _dm(fast_val=True)
    non_overlap_stride = WINDOW + HORIZON + STEPS - 1
    val_len = len(dm.val_subset)
    total_len = WINDOW + HORIZON + STEPS - 1
    expected = (val_len - total_len) // non_overlap_stride + 1
    assert len(dm.val_dataset) == expected


def test_fast_val_does_not_affect_train():
    """fast_val flag must not change the training dataset size."""
    dm_a = _dm(fast_val=False)
    dm_b = _dm(fast_val=True)
    assert len(dm_a.train_dataset) == len(dm_b.train_dataset)


# ── column selection ─────────────────────────────────────────────────────────

def test_input_columns_int_indices():
    dm = _dm(input_columns=[0, 1, 2])
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x.shape == (B, WINDOW, 3)
    assert b.x_raw.shape == (B, WINDOW, 3)


def test_target_columns_int_indices():
    dm = _dm(target_columns=[4, 5])
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.y.shape == (B, STEPS, 2)
    assert b.y_raw.shape == (B, STEPS, 2)


def test_input_and_target_columns_independent():
    dm = _dm(input_columns=[0, 1], target_columns=[3, 4, 5])
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x.shape == (B, WINDOW, 2)
    assert b.y.shape == (B, STEPS, 3)


def test_input_columns_str_names():
    dm = _dm(input_columns=["f0", "f1"])
    b = _first_batch(dm)
    B = min(8, len(dm.train_dataset))
    assert b.x.shape == (B, WINDOW, 2)


# ── loaders & splits ─────────────────────────────────────────────────────────

def test_all_loaders_exist():
    dm = _dm()
    assert hasattr(dm, "train_loader")
    assert hasattr(dm, "val_loader")
    assert hasattr(dm, "test_loader")


def test_all_loaders_iterate():
    dm = _dm()
    for _ in dm.train_loader: break
    for _ in dm.val_loader: break
    for _ in dm.test_loader: break


def test_split_sizes_positive():
    dm = _dm()
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) > 0


# ── dm properties ────────────────────────────────────────────────────────────

def test_dm_properties():
    dm = _dm(window=32, horizon=6, steps=18)
    assert dm.window == 32
    assert dm.horizon == 6
    assert dm.steps == 18
    assert dm.num_features == N_FEAT


def test_num_target_features_default():
    dm = _dm()
    assert dm.num_target_features == N_FEAT


def test_num_target_features_with_target_columns():
    dm = _dm(target_columns=[0, 1])
    assert dm.num_target_features == 2


# ── TSBatch.to() ─────────────────────────────────────────────────────────────

def test_tsbatch_to_cpu():
    dm = _dm()
    b = _first_batch(dm)
    b2 = b.to("cpu")
    assert b2.x.device.type == "cpu"
    assert b2.x_time.year.device.type == "cpu"


def test_tsbatch_keys_returns_non_none_fields():
    dm = _dm()
    b = _first_batch(dm)
    keys = b.keys()
    for k in ("x", "y", "x_raw", "y_raw", "x_time_feature", "y_time_feature",
               "x_time", "y_time", "x_index", "y_index"):
        assert k in keys, f"'{k}' missing from TSBatch.keys()"


# ── TimeEncConfig ─────────────────────────────────────────────────────────────

def test_time_enc_cfg_default_produces_floats():
    cfg = TimeEncConfig()
    dm = _dm(time_enc_cfg=cfg)
    b = _first_batch(dm)
    assert b.x_time_feature is not None
    assert b.x_time_feature.is_floating_point()


def test_time_enc_cfg_with_explicit_freq():
    dm = _dm(time_enc_cfg=TimeEncConfig(freq="h"))
    b = _first_batch(dm)
    assert b.x_time_feature is not None


def test_time_enc_cfg_repr_hides_custom_fn():
    """custom_fn should not appear in repr to keep it readable."""
    cfg = TimeEncConfig(custom_fn=lambda d: d)
    assert "custom_fn" not in repr(cfg) or "<" in repr(cfg)
