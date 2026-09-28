"""
多维下钻
职责3-智能归因：当核心指标波动时，沿维度逐层下钻定位贡献最大的切片。
支持：单维贡献度、逐层 drill-down、贡献度排序。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def dimension_contribution(
    df: pd.DataFrame,
    dim_col: str,
    metric_col: str,
    current_period: str,
    base_period: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    单维度贡献度分析：对比当期 vs 基期，计算各维度值的绝对贡献和贡献率。

    返回列：dim_value, base, current, delta, contribution_rate
    """
    cur = df[df[period_col] == current_period]
    base = df[df[period_col] == base_period]

    cur_agg = cur.groupby(dim_col)[metric_col].sum()
    base_agg = base.groupby(dim_col)[metric_col].sum()

    result = pd.DataFrame({"base": base_agg, "current": cur_agg}).fillna(0)
    result["delta"] = result["current"] - result["base"]
    total_delta = result["delta"].sum()
    result["contribution_rate"] = (
        result["delta"] / total_delta if total_delta != 0 else 0.0
    )
    result = result.sort_values("delta", key=abs, ascending=False).reset_index()
    result.columns = [dim_col, "base", "current", "delta", "contribution_rate"]
    return result


def multi_dim_drill_down(
    df: pd.DataFrame,
    dims: List[str],
    metric_col: str,
    current_period: str,
    base_period: str,
    period_col: str = "period",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    多维逐层下钻：从第一个维度开始，每轮取贡献 Top-N 切片继续下钻。
    返回下钻路径及各节点贡献度。
    """
    rows = []
    current_subset = df.copy()

    for level, dim in enumerate(dims):
        contrib = dimension_contribution(
            current_subset, dim, metric_col, current_period, base_period, period_col
        )
        top = contrib.head(top_n)
        for _, row in top.iterrows():
            rows.append(
                {
                    "level": level,
                    "dimension": dim,
                    "value": row[dim],
                    "base": row["base"],
                    "current": row["current"],
                    "delta": row["delta"],
                    "contribution_rate": row["contribution_rate"],
                }
            )
        # 下钻：只保留当前维度 Top-N 的值，进入下一层
        if level < len(dims) - 1:
            top_values = top[dim].tolist()
            current_subset = current_subset[
                current_subset[dim].isin(top_values)
            ]

    return pd.DataFrame(rows)


def detect_anomaly_period(
    df: pd.DataFrame,
    metric_col: str,
    period_col: str = "period",
    threshold: float = 0.3,
) -> Optional[tuple]:
    """
    自动检测异常波动期：找到环比变化超过 threshold 的连续两期。
    返回 (base_period, current_period) 或 None。
    """
    agg = df.groupby(period_col)[metric_col].sum().sort_index()
    if len(agg) < 2:
        return None
    pct = agg.pct_change().dropna()
    anomalies = pct[pct.abs() > threshold]
    if len(anomalies) == 0:
        return None
    # 取波动最大的一期
    target = anomalies.abs().idxmax()
    idx = agg.index.get_loc(target)
    base_period = agg.index[idx - 1]
    return (base_period, target)
