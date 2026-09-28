"""
时序预测模块
职责1-数据建模：基于历史销量预测未来，辅助业务制定目标。
支持：移动平均、指数平滑、ARIMA、Prophet（可选）、回归特征预测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    """预测结果封装"""
    history: pd.Series
    forecast: pd.Series
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    model_name: str = ""
    metrics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {"history": self.history, "forecast": self.forecast}
        )
        if self.lower is not None:
            df["lower"] = self.lower
        if self.upper is not None:
            df["upper"] = self.upper
        return df


def calc_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    """计算预测指标：MAPE / RMSE / MAE"""
    mask = actual.notna() & pred.notna()
    a, p = actual[mask], pred[mask]
    if len(a) == 0:
        return {}
    mape = float(np.mean(np.abs((a - p) / np.where(a == 0, 1, a))) * 100)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    return {"MAPE(%)": round(mape, 2), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}


class MovingAverageForecaster:
    """移动平均预测：简单平滑基线"""

    def __init__(self, window: int = 3):
        self.window = window

    def fit_predict(self, series: pd.Series, steps: int = 3) -> ForecastResult:
        history = series.copy()
        ma = history.rolling(self.window, min_periods=1).mean()
        last_val = float(ma.iloc[-1])
        future_idx = pd.date_range(
            history.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq=history.index.inferred_freq or "MS",
        )
        forecast = pd.Series([last_val] * steps, index=future_idx)
        metrics = calc_metrics(history, ma)
        return ForecastResult(
            history=history,
            forecast=forecast,
            model_name=f"MA({self.window})",
            metrics=metrics,
        )


class ExponentialSmoothingForecaster:
    """Holt-Winters 指数平滑，捕捉趋势与季节性"""

    def __init__(self, seasonal_periods: Optional[int] = None, trend: str = "add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend

    def fit_predict(self, series: pd.Series, steps: int = 3) -> ForecastResult:
        seasonal = "add" if self.seasonal_periods else None
        model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit()
        forecast = model.forecast(steps)
        fitted = model.fittedvalues
        metrics = calc_metrics(series, fitted)
        return ForecastResult(
            history=series,
            forecast=forecast,
            model_name="HoltWinters",
            metrics=metrics,
        )


class ARIMAForecaster:
    """ARIMA(p,d,q) 时序预测"""

    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order

    def fit_predict(self, series: pd.Series, steps: int = 3) -> ForecastResult:
        model = ARIMA(series, order=self.order).fit()
        forecast = model.forecast(steps)
        fitted = model.fittedvalues
        metrics = calc_metrics(series, fitted)
        return ForecastResult(
            history=series,
            forecast=forecast,
            model_name=f"ARIMA{self.order}",
            metrics=metrics,
        )


class ProphetForecaster:
    """
    Prophet 预测（可选依赖）。
    若未安装 prophet，调用时抛出 ImportError。
    """

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly = yearly_seasonality
        self.weekly = weekly_seasonality

    def fit_predict(self, series: pd.Series, steps: int = 30) -> ForecastResult:
        try:
            from prophet import Prophet
        except ImportError as e:
            raise ImportError("请先安装 prophet: pip install prophet") from e

        df = pd.DataFrame({"ds": series.index, "y": series.values})
        model = Prophet(
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            daily_seasonality=False,
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=steps)
        fcst = model.predict(future)
        forecast = fcst.set_index("ds")["yhat"].iloc[-steps:]
        lower = fcst.set_index("ds")["yhat_lower"].iloc[-steps:]
        upper = fcst.set_index("ds")["yhat_upper"].iloc[-steps:]
        fitted = fcst.set_index("ds")["yhat"].iloc[: len(series)]
        metrics = calc_metrics(series, pd.Series(fitted.values, index=series.index))
        return ForecastResult(
            history=series,
            forecast=forecast,
            lower=lower,
            upper=upper,
            model_name="Prophet",
            metrics=metrics,
        )


def compare_forecasters(series: pd.Series, steps: int = 3) -> pd.DataFrame:
    """对比多个预测器的效果，返回指标表"""
    results = []
    forecasters = [
        MovingAverageForecaster(window=3),
        ExponentialSmoothingForecaster(),
        ARIMAForecaster(order=(1, 1, 1)),
    ]
    for f in forecasters:
        try:
            res = f.fit_predict(series, steps=steps)
            results.append({"model": res.model_name, **res.metrics})
        except Exception as e:
            results.append({"model": f.__class__.__name__, "error": str(e)})
    return pd.DataFrame(results)
