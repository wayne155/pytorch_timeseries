import sys

import pytest


def test_cli_dispatches_registered_experiment(monkeypatch):
    from torch_timeseries.cli import exp as cli_exp
    from torch_timeseries.experiments import DLinearForecast

    captured = {}

    def fake_fire(target):
        captured["target"] = target

    monkeypatch.setattr(cli_exp.fire, "Fire", fake_fire)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytexp", "--model", "DLinear", "--task", "Forecast", "--help"],
    )

    cli_exp.exp()

    assert captured["target"] is DLinearForecast
    assert "--model" not in sys.argv
    assert "--task" not in sys.argv


def test_cli_rejects_unknown_experiment(monkeypatch):
    from torch_timeseries.cli import exp as cli_exp

    monkeypatch.setattr(
        sys,
        "argv",
        ["pytexp", "--model", "MissingModel", "--task", "Forecast"],
    )

    with pytest.raises(NotImplementedError, match="Unknown experiment"):
        cli_exp.exp()


def test_parse_type_uses_registry_without_evaluating_code():
    from torch_timeseries.utils.parse_type import parse_type

    class Expected:
        pass

    assert parse_type("Expected", {"Expected": Expected}) is Expected

    with pytest.raises(NotImplementedError):
        parse_type("__import__('os').system('echo unsafe')", {"Expected": Expected})


def test_time_encoding_enum_values_match_int_behaviour():
    import numpy as np
    import pandas as pd
    from torch_timeseries.utils.timefeatures import time_features, TimeEncoding

    dates = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="h")})

    out_int  = time_features(dates, timeenc=0, freq="h")
    out_enum = time_features(dates, timeenc=TimeEncoding.CALENDAR, freq="h")
    np.testing.assert_array_equal(out_int, out_enum)

    out_int1  = time_features(dates, timeenc=1, freq="h")
    out_enum1 = time_features(dates, timeenc=TimeEncoding.FOURIER, freq="h")
    np.testing.assert_array_equal(out_int1, out_enum1)
