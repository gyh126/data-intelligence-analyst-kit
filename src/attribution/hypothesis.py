"""
假设检验
职责3-智能归因：对归因结论进行统计显著性验证。
支持：t 检验、Mann-Whitney U、卡方检验、KS 检验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class HypothesisResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str

    def __str__(self):
        sig = "显著" if self.significant else "不显著"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4f} ({sig}) — {self.interpretation}"
        )


class HypothesisTester:
    """假设检验工具集"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def t_test(self, a: pd.Series, b: pd.Series) -> HypothesisResult:
        """独立样本 t 检验：比较两组均值是否有显著差异"""
        a, b = a.dropna(), b.dropna()
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        sig = bool(p < self.alpha)
        return HypothesisResult(
            test_name="Welch's t-test",
            statistic=float(stat),
            p_value=float(p),
            significant=sig,
            interpretation=(
                f"两组均值差异{'显著' if sig else '不显著'}"
                f"(α={self.alpha})"
            ),
        )

    def mann_whitney(self, a: pd.Series, b: pd.Series) -> HypothesisResult:
        """Mann-Whitney U 检验：非参数，比较两组分布是否相同"""
        a, b = a.dropna(), b.dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return HypothesisResult(
            test_name="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            significant=bool(p < self.alpha),
            interpretation=(
                f"两组分布差异{'显著' if p < self.alpha else '不显著'}"
            ),
        )

    def chi_square(self, observed: pd.DataFrame) -> HypothesisResult:
        """卡方检验：检验两个分类变量是否独立"""
        chi2, p, dof, _ = stats.chi2_contingency(observed)
        return HypothesisResult(
            test_name="Chi-square",
            statistic=float(chi2),
            p_value=float(p),
            significant=bool(p < self.alpha),
            interpretation=(
                f"分类变量关联性{'显著' if p < self.alpha else '不显著'}"
            ),
        )

    def ks_test(self, a: pd.Series, b: pd.Series) -> HypothesisResult:
        """KS 检验：比较两组分布是否一致"""
        a, b = a.dropna(), b.dropna()
        stat, p = stats.ks_2samp(a, b)
        return HypothesisResult(
            test_name="Kolmogorov-Smirnov",
            statistic=float(stat),
            p_value=float(p),
            significant=bool(p < self.alpha),
            interpretation=(
                f"两组分布差异{'显著' if p < self.alpha else '不显著'}"
            ),
        )

    def test_significance(
        self,
        df: pd.DataFrame,
        metric_col: str,
        group_col: str,
        group_a,
        group_b,
    ) -> pd.DataFrame:
        """
        对指定指标在两组间做多种检验，返回对比表。
        """
        a = df[df[group_col] == group_a][metric_col]
        b = df[df[group_col] == group_b][metric_col]
        results = [
            self.t_test(a, b),
            self.mann_whitney(a, b),
            self.ks_test(a, b),
        ]
        return pd.DataFrame(
            [
                {
                    "test": r.test_name,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "significant": r.significant,
                }
                for r in results
            ]
        )
