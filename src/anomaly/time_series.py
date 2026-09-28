"""
时序异常检测
职责2-异常检测：针对时间序列的专用检测方法。
支持：滚动统计、STL 分解残差、CUSUM 累积和。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TimeSeriesAnomalyDetector:
    """时间序列异常检测"""

    def __init__(self, method: str = "rolling", **kwargs):
        """
        method: rolling | stl_residual | cusum
        """
        self.method = method
        self.kwargs = kwargs

    def detect(self, series: pd.Series) -> pd.DataFrame:
        s = series.dropna()
        if self.method == "rolling":
            return self._rolling(s)
        if self.method == "stl_residual":
            return self._stl_residual(s)
        if self.method == "cusum":
            return self._cusum(s)
        raise ValueError(f"未知方法: {self.method}")

    def _rolling(self, s: pd.Series) -> pd.DataFrame:
        """滚动窗口均值±kσ"""
        window = self.kwargs.get("window", 7)
        k = self.kwargs.get("k", 3)
        roll_mean = s.rolling(window, min_periods=1).mean()
        roll_std = s.rolling(window, min_periods=1).std().fillna(0)
        upper = roll_mean + k * roll_std
        lower = roll_mean - k * roll_std
        is_anom = (s > upper) | (s < lower)
        return pd.DataFrame(
            {
                "value": s,
                "roll_mean": roll_mean,
                "upper": upper,
                "lower": lower,
                "is_anomaly": is_anom,
            }
        )

    def _stl_residual(self, s: pd.Series) -> pd.DataFrame:
        """STL 分解后对残差做 3σ 检测"""
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError as e:
            raise ImportError("需要 statsmodels") from e
        period = self.kwargs.get("period", 7)
        stl = STL(s, period=period, robust=True).fit()
        resid = stl.resid
        mu, std = resid.mean(), resid.std()
        k = self.kwargs.get("k", 3)
        is_anom = (resid - mu).abs() > k * std
        return pd.DataFrame(
            {
                "value": s,
                "trend": stl.trend,
                "seasonal": stl.seasonal,
                "residual": resid,
                "is_anomaly": is_anom,
            }
        )

    def _cusum(self, s: pd.Series) -> pd.DataFrame:
        """
        CUSUM 累积和控制图，检测均值漂移。
        """
        target = self.kwargs.get("target", s.mean())
        std = self.kwargs.get("std", s.std())
        k = self.kwargs.get("k", 0.5)
        h = self.kwargs.get("h", 5.0)

        dev = (s - target) / (std if std > 0 else 1)
        pos = np.zeros(len(s))
        neg = np.zeros(len(s))
        for i in range(1, len(s)):
            pos[i] = max(0, pos[i - 1] + dev.iloc[i] - k)
            neg[i] = min(0, neg[i - 1] + dev.iloc[i] + k)

        is_anom = (pos > h) | (neg < -h)
        return pd.DataFrame(
            {
                "value": s,
                "cusum_pos": pos,
                "cusum_neg": neg,
                "is_anomaly": is_anom,
            }
        )
