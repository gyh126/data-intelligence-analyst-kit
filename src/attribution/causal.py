"""
因果推断模块（JD 加分项）
职责3-智能归因：超越相关性，识别指标波动的因果根因。
支持：Granger 因果检验、DoWhy 因果效应估计。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CausalResult:
    method: str
    treatment: str
    outcome: str
    effect: float
    p_value: Optional[float]
    interpretation: str

    def __str__(self):
        p = f"p={self.p_value:.4f}" if self.p_value is not None else "p=N/A"
        return f"[{self.method}] {self.treatment} → {self.outcome}: effect={self.effect:.4f}, {p}. {self.interpretation}"


class GrangerCausality:
    """
    Granger 因果检验：判断一个时间序列是否对另一个有预测作用。
    注意：Granger 因果不等于真实因果，但可作为时序归因的线索。
    """

    def __init__(self, maxlag: int = 5, alpha: float = 0.05):
        self.maxlag = maxlag
        self.alpha = alpha

    def test(self, df: pd.DataFrame, target: str, candidate: str) -> CausalResult:
        """
        检验 candidate 是否 Granger 导致 target。
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError as e:
            raise ImportError("需要 statsmodels") from e

        data = df[[candidate, target]].dropna()
        if len(data) < 3 * self.maxlag:
            return CausalResult(
                method="Granger",
                treatment=candidate,
                outcome=target,
                effect=0.0,
                p_value=None,
                interpretation="样本量不足，无法进行 Granger 检验",
            )

        # statsmodels 新版本不再支持 verbose 参数，直接调用
        try:
            result = grangercausalitytests(data, maxlag=self.maxlag, verbose=False)
        except TypeError:
            result = grangercausalitytests(data, maxlag=self.maxlag)
        # 取最小 p 值
        p_values = []
        for lag, res in result.items():
            # res[0] 是各检验统计量 dict
            ssr_ftest = res[0]["ssr_ftest"]
            p_values.append(ssr_ftest[1])
        min_p = min(p_values)
        best_lag = p_values.index(min_p) + 1

        return CausalResult(
            method="Granger",
            treatment=candidate,
            outcome=target,
            effect=float(best_lag),
            p_value=float(min_p),
            interpretation=(
                f"在 lag={best_lag} 时 p={min_p:.4f}，"
                f"{'存在' if min_p < self.alpha else '不存在'}显著 Granger 因果"
            ),
        )

    def test_all(
        self, df: pd.DataFrame, target: str, candidates: list
    ) -> pd.DataFrame:
        """检验多个候选变量对 target 的 Granger 因果"""
        rows = []
        for c in candidates:
            try:
                r = self.test(df, target, c)
                rows.append(
                    {
                        "candidate": c,
                        "best_lag": r.effect,
                        "p_value": r.p_value,
                        "significant": r.p_value < self.alpha
                        if r.p_value is not None
                        else False,
                    }
                )
            except Exception as e:
                rows.append({"candidate": c, "error": str(e)})
        return pd.DataFrame(rows)


class DoWhyCausalEstimator:
    """
    DoWhy 因果效应估计：通过因果图 + 反事实估计处理变量对结果的因果效应。
    依赖 dowhy 库（可选）。
    """

    def __init__(self, treatment: str, outcome: str, common_causes: list):
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes

    def estimate(
        self,
        df: pd.DataFrame,
        method: str = "backdoor.linear_regression",
    ) -> CausalResult:
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError as e:
            raise ImportError("请先安装 dowhy: pip install dowhy") from e

        model = CausalModel(
            data=df,
            treatment=self.treatment,
            outcome=self.outcome,
            common_causes=self.common_causes,
        )
        identified = model.identify_effect()
        estimate = model.estimate_effect(
            identified_estimand=identified, method_name=method
        )
        # 反驳检验
        try:
            refute = model.refute_estimate(
                identified, estimate, method_name="placebo_treatment_refuter"
            )
            p_value = float(refute.refutation_result["p_value"])
        except Exception:
            p_value = None

        return CausalResult(
            method="DoWhy",
            treatment=self.treatment,
            outcome=self.outcome,
            effect=float(estimate.value),
            p_value=p_value,
            interpretation=(
                f"因果效应估计值={estimate.value:.4f}，"
                f"安慰剂检验 p={p_value}"
                if p_value is not None
                else f"因果效应估计值={estimate.value:.4f}"
            ),
        )
