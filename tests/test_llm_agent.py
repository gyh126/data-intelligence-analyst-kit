"""LLM Agent 模块测试"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_agent import (  # noqa: E402
    AnalysisAgent,
    MockLLMProvider,
    ReportGenerator,
    get_provider,
)


def test_mock_provider():
    llm = get_provider("mock")
    assert llm.is_available()
    resp = llm.chat("测试")
    assert len(resp.content) > 0
    assert resp.model == "mock-llm-v1"


def test_report_generator():
    reporter = ReportGenerator(MockLLMProvider())
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = reporter.generate_exploration_report(df)
    assert len(report) > 50


def test_analysis_agent():
    df = pd.DataFrame(
        {
            "month": pd.date_range("2026-01-01", periods=6, freq="MS").tolist() * 2,
            "region": ["华东"] * 6 + ["华南"] * 6,
            "sales": [100, 110, 105, 120, 130, 125, 90, 85, 95, 88, 92, 90],
        }
    )
    df["period"] = df["month"].dt.strftime("%Y-%m")
    agent = AnalysisAgent(
        df=df, metric_col="sales", period_col="period", time_col="month"
    )
    report = agent.run_all()
    assert isinstance(report, str)
    summary = agent.summary()
    assert len(summary) == 5  # 5 个步骤
