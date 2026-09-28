"""
机器学习异常检测
职责2-异常检测：用 ML 算法从多维特征中发现异常模式。
支持：Isolation Forest、One-Class SVM、XGBoost（有监督/半监督）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class MLAnomalyDetector:
    """
    多维异常检测。
    method: isolation_forest | one_class_svm | xgboost
    """

    def __init__(self, method: str = "isolation_forest", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        self.model = None

    def _build_model(self):
        if self.method == "isolation_forest":
            return IsolationForest(
                n_estimators=self.kwargs.get("n_estimators", 200),
                contamination=self.kwargs.get("contamination", 0.05),
                random_state=42,
                n_jobs=-1,
            )
        if self.method == "one_class_svm":
            return OneClassSVM(
                nu=self.kwargs.get("nu", 0.05),
                kernel="rbf",
                gamma="scale",
            )
        if self.method == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError("请先安装 xgboost") from e
            return XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
            )
        raise ValueError(f"未知方法: {self.method}")

    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        无监督方法（IF / OCSVM）直接 fit_predict。
        XGBoost 需要有标签，可通过 fit_supervised 调用。
        """
        Xs = self.scaler.fit_transform(X)
        self.model = self._build_model()
        if self.method == "xgboost":
            raise ValueError("XGBoost 为有监督方法，请使用 fit_supervised")
        preds = self.model.fit_predict(Xs)
        scores = -self.model.score_samples(Xs)
        is_anom = preds == -1
        return pd.DataFrame(
            {"anomaly_score": scores, "is_anomaly": is_anom}, index=X.index
        )

    def fit_supervised(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> pd.DataFrame:
        """有监督异常检测（XGBoost），y 为 0/1 标签"""
        if self.method != "xgboost":
            raise ValueError("fit_supervised 仅支持 xgboost")
        Xs = self.scaler.fit_transform(X)
        self.model = self._build_model()
        self.model.fit(Xs, y)
        proba = self.model.predict_proba(Xs)[:, 1]
        return pd.DataFrame(
            {"anomaly_prob": proba, "is_anomaly": proba >= threshold},
            index=X.index,
        )
