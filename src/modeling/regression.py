"""
回归评估模型
职责1-数据建模：建立销量评估模型，量化各因素对业务指标的贡献，指导目标设定。
支持：线性回归、随机森林、XGBoost，输出特征重要性与可解释性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionResult:
    model_name: str
    r2: float
    rmse: float
    mae: float
    feature_importance: pd.Series
    model: object

    def summary(self) -> str:
        return (
            f"[{self.model_name}] R2={self.r2:.4f} RMSE={self.rmse:.2f} "
            f"MAE={self.mae:.2f}\nTop features:\n{self.feature_importance.head(10)}"
        )


class RegressionEvaluator:
    """
    统一回归评估器。
    支持 linear / ridge / random_forest / xgboost。
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.scaler = StandardScaler()

    def _build_model(self):
        if self.model_type == "linear":
            return LinearRegression()
        if self.model_type == "ridge":
            return Ridge(alpha=self.kwargs.get("alpha", 1.0))
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.kwargs.get("n_estimators", 200),
                max_depth=self.kwargs.get("max_depth", 8),
                random_state=42,
                n_jobs=-1,
            )
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as e:
                raise ImportError("请先安装 xgboost: pip install xgboost") from e
            return XGBRegressor(
                n_estimators=self.kwargs.get("n_estimators", 300),
                max_depth=self.kwargs.get("max_depth", 6),
                learning_rate=self.kwargs.get("learning_rate", 0.05),
                random_state=42,
                n_jobs=-1,
            )
        raise ValueError(f"未知模型类型: {self.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> RegressionResult:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = self._build_model()
        # 线性模型需要标准化
        if self.model_type in ("linear", "ridge"):
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))

        # 特征重要性
        if self.model_type in ("linear", "ridge"):
            importance = pd.Series(
                np.abs(model.coef_), index=X.columns
            ).sort_values(ascending=False)
        else:
            importance = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        return RegressionResult(
            model_name=self.model_type,
            r2=float(r2),
            rmse=rmse,
            mae=mae,
            feature_importance=importance,
            model=model,
        )


def compare_models(
    X: pd.DataFrame, y: pd.Series, models: Optional[list] = None
) -> pd.DataFrame:
    """对比多个回归模型"""
    if models is None:
        models = ["linear", "ridge", "random_forest", "xgboost"]
    rows = []
    for m in models:
        try:
            res = RegressionEvaluator(model_type=m).fit(X, y)
            rows.append(
                {
                    "model": m,
                    "R2": round(res.r2, 4),
                    "RMSE": round(res.rmse, 2),
                    "MAE": round(res.mae, 2),
                    "top_feature": res.feature_importance.index[0]
                    if len(res.feature_importance)
                    else "",
                }
            )
        except Exception as e:
            rows.append({"model": m, "error": str(e)})
    return pd.DataFrame(rows)
