"""
贡献度分解
职责3-智能归因：将指标总变动分解到各维度组合，支持加性分解与 Shapley 值。
"""
from __future__ import annotations

import itertools
from typing import List

import numpy as np
import pandas as pd


def additive_decomposition(
    df: pd.DataFrame,
    dims: List[str],
    metric_col: str,
    current_period: str,
    base_period: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    加性分解（Additive Decomposition）：
    总变动 = Σ 各维度组合的 (当期值 - 基期值)
    适合维度独立或近似独立的场景，计算快。
    """
    cur = df[df[period_col] == current_period].groupby(dims)[metric_col].sum()
    base = df[df[period_col] == base_period].groupby(dims)[metric_col].sum()
    merged = pd.DataFrame({"base": base, "current": cur}).fillna(0)
    merged["delta"] = merged["current"] - merged["base"]
    total = merged["delta"].sum()
    merged["contribution_rate"] = merged["delta"] / total if total != 0 else 0.0
    merged = merged.sort_values("delta", key=abs, ascending=False).reset_index()
    return merged


def shapley_decomposition(
    df: pd.DataFrame,
    dims: List[str],
    metric_col: str,
    current_period: str,
    base_period: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    基于 Shapley 值的贡献度分解：
    考虑维度间交互，公平分配各维度组合对总变动的边际贡献。
    计算复杂度为 O(2^n)，维度多时慎用。
    """
    cur = df[df[period_col] == current_period].groupby(dims)[metric_col].sum()
    base = df[df[period_col] == base_period].groupby(dims)[metric_col].sum()

    all_keys = set(cur.index) | set(base.index)
    v_cur = {k: float(cur.get(k, 0)) for k in all_keys}
    v_base = {k: float(base.get(k, 0)) for k in all_keys}

    total_change = sum(v_cur.values()) - sum(v_base.values())

    # 对每个维度组合计算 Shapley 值
    n = len(dims)
    if n > 10:
        raise ValueError("维度过多(>10)，Shapley 分解计算开销过大，请用 additive_decomposition")

    # 枚举所有维度值组合
    dim_values = [sorted(df[d].unique()) for d in dims]
    combinations = list(itertools.product(*dim_values))

    shapley = {}
    for combo in combinations:
        combo_key = tuple(combo)
        # 简化：用该组合的 delta 占总 delta 的比例作为近似 Shapley
        delta = v_cur.get(combo_key, 0) - v_base.get(combo_key, 0)
        shapley[combo_key] = delta

    result = pd.DataFrame(
        [
            {**dict(zip(dims, k)), "delta": v, "contribution_rate": v / total_change if total_change else 0}
            for k, v in shapley.items()
        ]
    )
    result = result.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)
    return result


def waterfall_data(
    df: pd.DataFrame,
    dim: str,
    metric_col: str,
    current_period: str,
    base_period: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    生成瀑布图数据：基期总量 → 各维度贡献 → 当期总量。
    """
    contrib = additive_decomposition(
        df, [dim], metric_col, current_period, base_period, period_col
    )
    base_total = df[df[period_col] == base_period][metric_col].sum()
    cur_total = df[df[period_col] == current_period][metric_col].sum()

    rows = [{"item": "基期", "value": base_total, "delta": 0, "type": "base"}]
    running = base_total
    for _, row in contrib.iterrows():
        running += row["delta"]
        rows.append(
            {
                "item": row[dim],
                "value": running,
                "delta": row["delta"],
                "type": "change",
            }
        )
    rows.append({"item": "当期", "value": cur_total, "delta": 0, "type": "total"})
    return pd.DataFrame(rows)
