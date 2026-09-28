"""
聚类分析模块
职责1-数据建模：对产品/区域/客户进行分群，识别业务结构，辅助制定差异化目标。
支持：KMeans、层次聚类、DBSCAN，含轮廓系数评估与最优 K 选择。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    def __init__(self, method: str = "kmeans", n_clusters: int = 4, **kwargs):
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.scaler = StandardScaler()

    def _build_model(self):
        if self.method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10
            )
        if self.method == "hierarchical":
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        if self.method == "dbscan":
            return DBSCAN(
                eps=self.kwargs.get("eps", 0.5),
                min_samples=self.kwargs.get("min_samples", 5),
            )
        raise ValueError(f"未知聚类方法: {self.method}")

    def fit(self, df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
        """对指定列聚类，返回带 cluster 标签的副本"""
        data = df[cols] if cols else df.select_dtypes(include=[np.number])
        X = self.scaler.fit_transform(data)
        model = self._build_model()
        labels = model.fit_predict(X)
        out = df.copy()
        out["cluster"] = labels
        return out

    @staticmethod
    def silhouette_at_k(
        df: pd.DataFrame, cols: list, k_range: range = range(2, 11)
    ) -> pd.DataFrame:
        """不同 K 值下的轮廓系数，辅助选 K"""
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cols])
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            rows.append({"k": k, "silhouette": round(score, 4)})
        return pd.DataFrame(rows)

    @staticmethod
    def cluster_profile(
        df: pd.DataFrame, label_col: str = "cluster"
    ) -> pd.DataFrame:
        """输出各簇在数值特征上的均值画像"""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in numeric:
            numeric.remove(label_col)
        return df.groupby(label_col)[numeric].mean().round(2)
