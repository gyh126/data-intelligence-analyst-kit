"""
分析 Agent 编排
职责4-AI 探索：端到端自动化分析工作流。
数据 → 探索 → 建模 → 异常检测 → 归因 → 报告。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from anomaly import StatisticalAnomalyDetector, TimeSeriesAnomalyDetector
from attribution import (
    HypothesisTester,
    additive_decomposition,
    detect_anomaly_period,
    dimension_contribution,
)
from modeling import ARIMAForecaster, RegressionEvaluator
from llm_agent.provider import BaseLLMProvider, MockLLMProvider
from llm_agent.report_generator import ReportGenerator


@dataclass
class AgentStep:
    name: str
    status: str = "pending"
    result: Optional[object] = None
    message: str = ""


@dataclass
class AnalysisAgent:
    """
    端到端数据分析 Agent。
    输入数据，自动执行：探索 → 建模 → 异常 → 归因 → 报告。
    """

    df: pd.DataFrame
    metric_col: str
    period_col: str = "period"
    time_col: Optional[str] = None
    dim_cols: Optional[list] = None
    llm: Optional[BaseLLMProvider] = None
    steps: list = field(default_factory=list)

    def __post_init__(self):
        if self.llm is None:
            self.llm = MockLLMProvider()
        self.reporter = ReportGenerator(self.llm)
        if self.time_col is None:
            # 自动推断时间列
            for c in self.df.columns:
                if "date" in c.lower() or "month" in c.lower() or "time" in c.lower():
                    self.time_col = c
                    break
        if self.dim_cols is None:
            self.dim_cols = [
                c
                for c in self.df.columns
                if c not in [self.metric_col, self.period_col, self.time_col]
                and self.df[c].dtype == "object"
            ]

    def _run_step(self, name: str, func, **kwargs) -> AgentStep:
        step = AgentStep(name=name)
        try:
            step.result = func(**kwargs)
            step.status = "success"
            step.message = f"{name} 完成"
        except Exception as e:
            step.status = "failed"
            step.message = f"{name} 失败: {e}"
        self.steps.append(step)
        return step

    def run_exploration(self) -> AgentStep:
        """步骤1：数据探索"""
        return self._run_step(
            "数据探索",
            self.reporter.generate_exploration_report,
            df=self.df,
        )

    def run_modeling(self) -> AgentStep:
        """步骤2：数据建模（时序预测）"""
        def _forecast():
            series = self.df.groupby(self.time_col)[self.metric_col].sum()
            forecaster = ARIMAForecaster(order=(1, 1, 1))
            res = forecaster.fit_predict(series, steps=3)
            return res

        return self._run_step("数据建模", _forecast)

    def run_anomaly_detection(self) -> AgentStep:
        """步骤3：异常检测"""
        def _detect():
            series = self.df.groupby(self.time_col)[self.metric_col].sum()
            detector = TimeSeriesAnomalyDetector(method="rolling", window=3, k=3)
            return detector.detect(series)

        return self._run_step("异常检测", _detect)

    def run_attribution(self) -> AgentStep:
        """步骤4：智能归因"""
        def _attribute():
            periods = sorted(self.df[self.period_col].unique())
            if len(periods) < 2:
                return None
            base, current = periods[-2], periods[-1]
            results = {}
            for dim in (self.dim_cols or [])[:3]:
                results[dim] = dimension_contribution(
                    self.df, dim, self.metric_col, current, base, self.period_col
                )
            return results

        return self._run_step("智能归因", _attribute)

    def run_report(self) -> AgentStep:
        """步骤5：LLM 生成综合报告"""
        def _report():
            parts = []
            for step in self.steps:
                if step.status == "success" and step.result is not None:
                    if isinstance(step.result, str):
                        parts.append(step.result)
            return "\n\n---\n\n".join(parts) if parts else "无可用分析结果"

        return self._run_step("综合报告", _report)

    def run_all(self) -> str:
        """执行完整分析流程，返回综合报告"""
        self.run_exploration()
        self.run_modeling()
        self.run_anomaly_detection()
        self.run_attribution()
        report_step = self.run_report()
        return report_step.result or "分析完成"

    def summary(self) -> pd.DataFrame:
        """返回各步骤执行状态"""
        return pd.DataFrame(
            [
                {"step": s.name, "status": s.status, "message": s.message}
                for s in self.steps
            ]
        )
