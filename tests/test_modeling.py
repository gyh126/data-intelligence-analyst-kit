"""建模模块测试"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modeling import (  # noqa: E402
    ARIMAForecaster,
    ClusterAnalyzer,
    ExponentialSmoothingForecaster,
    MovingAverageForecaster,
    RegressionEvaluator,
    compare_forecasters,
)


@pytest.fixture
def series():
    return pd.Series(
        np.arange(100) + np.random.normal(0, 5, 100),
        index=pd.date_range("2025-01-01", periods=100, freq="D"),
    )


def test_moving_average(series):
    res = MovingAverageForecaster(window=3).fit_predict(series, steps=3)
    assert len(res.forecast) == 3
    assert "MAPE(%)" in res.metrics


def test_exponential_smoothing(series):
    res = ExponentialSmoothingForecaster().fit_predict(series, steps=3)
    assert len(res.forecast) == 3


def test_arima(series):
    res = ARIMAForecaster((1, 1, 1)).fit_predict(series, steps=3)
    assert len(res.forecast) == 3


def test_compare_forecasters(series):
    df = compare_forecasters(series, steps=3)
    assert len(df) >= 2
    assert "model" in df.columns


def test_regression():
    X = pd.DataFrame({"x1": np.random.rand(100), "x2": np.random.rand(100)})
    y = X["x1"] * 2 + X["x2"] * 3 + np.random.normal(0, 0.1, 100)
    res = RegressionEvaluator("random_forest").fit(X, y)
    assert res.r2 > 0.5
    assert len(res.feature_importance) == 2


def test_clustering():
    df = pd.DataFrame(
        {
            "x": np.concatenate([np.random.randn(50), np.random.randn(50) + 10]),
            "y": np.concatenate([np.random.randn(50), np.random.randn(50) + 10]),
        }
    )
    clustered = ClusterAnalyzer("kmeans", n_clusters=2).fit(df, ["x", "y"])
    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 2
