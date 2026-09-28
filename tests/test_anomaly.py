"""异常检测模块测试"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly import (  # noqa: E402
    MLAnomalyDetector,
    StatisticalAnomalyDetector,
    TimeSeriesAnomalyDetector,
)


@pytest.fixture
def series_with_anomaly():
    s = pd.Series(np.random.normal(100, 10, 100))
    s.iloc[50] = 500  # 注入异常
    return s


def test_sigma(series_with_anomaly):
    det = StatisticalAnomalyDetector("sigma", k=2).detect(series_with_anomaly)
    assert "is_anomaly" in det.columns
    assert det["is_anomaly"].any()


def test_iqr(series_with_anomaly):
    det = StatisticalAnomalyDetector("iqr").detect(series_with_anomaly)
    assert det["is_anomaly"].any()


def test_zscore(series_with_anomaly):
    det = StatisticalAnomalyDetector("zscore", threshold=2).detect(series_with_anomaly)
    assert det["is_anomaly"].any()


def test_esd(series_with_anomaly):
    det = StatisticalAnomalyDetector("esd", max_outliers=3).detect(series_with_anomaly)
    assert det["is_anomaly"].any()


def test_rolling_ts():
    s = pd.Series(
        np.random.normal(100, 5, 50),
        index=pd.date_range("2025-01-01", periods=50, freq="D"),
    )
    s.iloc[25] = 300
    det = TimeSeriesAnomalyDetector("rolling", window=5, k=2).detect(s)
    assert "is_anomaly" in det.columns


def test_isolation_forest():
    X = pd.DataFrame(np.random.randn(200, 3))
    X.iloc[0] = [10, 10, 10]  # 异常
    det = MLAnomalyDetector("isolation_forest", contamination=0.01).fit_predict(X)
    assert "is_anomaly" in det.columns
