import numpy as np
import pandas as pd

from precision.precision.controls import fourier_seasonality, holiday_flags, stack_controls


def test_fourier_seasonality_shapes_and_index():
    index = pd.date_range("2024-01-01", periods=4, freq="W")
    df = fourier_seasonality(4, period=4, K=2, index=index)

    assert list(df.columns) == ["sin_4_1", "cos_4_1", "sin_4_2", "cos_4_2"]
    assert df.index.equals(index)

    # sin/cos pairs are orthogonal across time for period=4
    dot = float(np.dot(df["sin_4_1"], df["cos_4_1"]))
    assert abs(dot) < 1e-9


def test_holiday_flags_handles_missing_weeks():
    index = pd.RangeIndex(5, name="time")
    spans = {"sale": [0, 2, 4], "brand": [-1, 10]}
    flags = holiday_flags(index, spans)

    assert list(flags.columns) == ["brand", "sale"]
    assert flags.loc[0, "sale"] == 1.0
    assert flags.loc[1:, "sale"].sum() == 2.0
    assert flags["brand"].sum() == 0.0


def test_stack_controls_standardize_and_alignment():
    idx = pd.RangeIndex(3, name="t")
    first = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    second = pd.DataFrame({"b": [10.0, np.nan, 30.0]}, index=idx)

    stacked = stack_controls(first, second)

    assert stacked.shape == (3, 2)
    # Standardisation yields zero mean and unit variance for each column.
    np.testing.assert_allclose(stacked.mean(axis=0), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(stacked.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-9)


def test_stack_controls_empty_returns_zero_matrix():
    out = stack_controls()
    assert out.shape == (0, 0)
