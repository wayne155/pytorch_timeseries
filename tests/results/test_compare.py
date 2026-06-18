"""Tests for ResultsComparator."""
import math
import pytest

from torch_timeseries.results import ResultsComparator, RunResult


def _make(model, dataset, task, seed, mse, mae):
    return RunResult(
        model=model, task=task, dataset=dataset, seed=seed,
        timestamp="2026-01-01T00:00:00",
        hparams={"lr": 1e-3},
        metrics={"mse": mse, "mae": mae},
        num_params=1_000_000,
        train_time_sec=60.0,
        git_commit="abc",
    )


RESULTS = [
    _make("DLinear",  "ETTh1", "Forecast", 1, 0.42, 0.45),
    _make("DLinear",  "ETTh1", "Forecast", 2, 0.43, 0.46),
    _make("DLinear",  "ETTh1", "Forecast", 3, 0.41, 0.44),
    _make("PatchTST", "ETTh1", "Forecast", 1, 0.38, 0.41),
    _make("PatchTST", "ETTh1", "Forecast", 2, 0.39, 0.42),
    _make("PatchTST", "ETTh1", "Forecast", 3, 0.37, 0.40),
    _make("PatchTST", "ETTm1", "Forecast", 1, 0.35, 0.38),
]


def test_summary_keys():
    cmp = ResultsComparator(RESULTS)
    s = cmp.summary()
    assert ("DLinear", "ETTh1", "Forecast") in s
    assert ("PatchTST", "ETTh1", "Forecast") in s


def test_summary_n_seeds():
    cmp = ResultsComparator(RESULTS)
    s = cmp.summary()
    assert int(s[("DLinear", "ETTh1", "Forecast")]["n_seeds"][0]) == 3
    assert int(s[("PatchTST", "ETTm1", "Forecast")]["n_seeds"][0]) == 1


def test_summary_mean_correct():
    cmp = ResultsComparator(RESULTS)
    s = cmp.summary()
    mean_mse, _ = s[("DLinear", "ETTh1", "Forecast")]["mse"]
    assert math.isclose(mean_mse, (0.42 + 0.43 + 0.41) / 3, rel_tol=1e-5)


def test_summary_std_single_seed_is_zero():
    cmp = ResultsComparator(RESULTS)
    s = cmp.summary()
    _, std = s[("PatchTST", "ETTm1", "Forecast")]["mse"]
    assert std == 0.0


def test_best_model():
    cmp = ResultsComparator(RESULTS)
    best = cmp.best_model("mse", dataset="ETTh1", task="Forecast")
    assert best == "PatchTST"


def test_best_model_higher_is_better():
    # invert: higher mae is "better" (nonsensical but tests the flag)
    cmp = ResultsComparator(RESULTS)
    worst = cmp.best_model("mae", dataset="ETTh1", task="Forecast", lower_is_better=False)
    assert worst == "DLinear"


def test_best_model_no_match_returns_none():
    cmp = ResultsComparator(RESULTS)
    assert cmp.best_model("mse", dataset="NonExistent") is None


def test_group_by_task_false():
    cmp = ResultsComparator(RESULTS, group_by_task=False)
    s = cmp.summary()
    assert ("DLinear", "ETTh1") in s
    # PatchTST ETTh1 should pool 3 ETTh1 seeds (still 3, not mixed with ETTm1)
    assert int(s[("PatchTST", "ETTh1")]["n_seeds"][0]) == 3


def test_metric_names_filtered():
    cmp = ResultsComparator(RESULTS, metrics=["mse"])
    assert cmp.metric_names == ["mse"]


def test_empty_results():
    cmp = ResultsComparator([])
    assert cmp.summary() == {}
    assert cmp.metric_names == []


def test_to_dataframe():
    pytest.importorskip("pandas")
    cmp = ResultsComparator(RESULTS)
    df = cmp.to_dataframe()
    assert "mse_mean" in df.columns
    assert "mse_std" in df.columns
    assert len(df) == len(cmp.summary())


def test_print_table_runs(capsys):
    cmp = ResultsComparator(RESULTS)
    cmp.print_table()
    captured = capsys.readouterr()
    assert "DLinear" in captured.out
    assert "PatchTST" in captured.out
    assert "mse" in captured.out
