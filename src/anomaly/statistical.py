"""
统计异常检测
职责2-异常检测：用经典统计方法发现异常点，作为人工经验规则的算法化升级。
支持：3σ 准则、IQR（箱线图）、Z-score、广义 ESD 检验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnomalyDetector:
    """统计异常检测，支持多种方法"""

    def __init__(self, method: str = "sigma", **kwargs):
        """
        method: sigma | iqr | zscore | esd
        kwargs: threshold / k / max_outliers / alpha
        """
        self.method = method
        self.kwargs = kwargs

    def detect(self, series: pd.Series) -> pd.DataFrame:
        """返回含 is_anomaly / score / bound 信息的 DataFrame"""
        s = series.dropna()
        if self.method == "sigma":
            return self._sigma(s)
        if self.method == "iqr":
            return self._iqr(s)
        if self.method == "zscore":
            return self._zscore(s)
        if self.method == "esd":
            return self._esd(s)
        raise ValueError(f"未知方法: {self.method}")

    def _sigma(self, s: pd.Series) -> pd.DataFrame:
        """3σ 准则"""
        mu, std = s.mean(), s.std()
        k = self.kwargs.get("k", 3)
        lower, upper = mu - k * std, mu + k * std
        is_anom = (s < lower) | (s > upper)
        return pd.DataFrame(
            {
                "value": s,
                "score": (s - mu) / std,
                "lower": lower,
                "upper": upper,
                "is_anomaly": is_anom,
            }
        )

    def _iqr(self, s: pd.Series) -> pd.DataFrame:
        """IQR 箱线图法"""
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        k = self.kwargs.get("k", 1.5)
        lower, upper = q1 - k * iqr, q3 + k * iqr
        is_anom = (s < lower) | (s > upper)
        return pd.DataFrame(
            {
                "value": s,
                "score": (s - s.median()) / iqr,
                "lower": lower,
                "upper": upper,
                "is_anomaly": is_anom,
            }
        )

    def _zscore(self, s: pd.Series) -> pd.DataFrame:
        """Z-score 阈值法"""
        z = np.abs(stats.zscore(s))
        threshold = self.kwargs.get("threshold", 3.0)
        is_anom = z > threshold
        return pd.DataFrame(
            {
                "value": s,
                "score": z,
                "is_anomaly": is_anom,
            }
        )

    def _esd(self, s: pd.Series) -> pd.DataFrame:
        """
        广义 ESD (Generalized Extreme Studentized Deviate) 检验。
        迭代检测多个异常点，适合近似正态分布数据。
        """
        max_outliers = self.kwargs.get("max_outliers", 5)
        alpha = self.kwargs.get("alpha", 0.05)
        values = s.values.copy()
        n = len(values)
        anomaly_idx = set()

        for i in range(max_outliers):
            if len(values) <= 3:
                break
            mean = values.mean()
            std = values.std(ddof=1)
            if std == 0:
                break
            deviations = np.abs(values - mean)
            max_dev_idx = np.argmax(deviations)
            R = deviations[max_dev_idx] / std
            # 临界值
            p = 1 - alpha / (2 * (n - i))
            t_crit = stats.t.ppf(p, n - i - 2)
            lam = (
                (n - i - 1)
                * t_crit
                / np.sqrt((n - i - 2 + t_crit**2) * (n - i))
            )
            if R > lam:
                anomaly_idx.add(int(np.where(s.values == values[max_dev_idx])[0][0]))
                values = np.delete(values, max_dev_idx)
            else:
                break

        is_anom = pd.Series(False, index=s.index)
        for idx in anomaly_idx:
            is_anom.iloc[idx] = True
        return pd.DataFrame(
            {
                "value": s,
                "is_anomaly": is_anom.values,
            }
        )
