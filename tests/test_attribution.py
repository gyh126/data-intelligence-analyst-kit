"""归因模块测试"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attribution import (  # noqa: E402
    GrangerCausality,
    HypothesisTester,
    additive_decomposition,
    detect_anomaly_period,
    dimension_contribution,
    multi_dim_drill_down,
    waterfall_data,
)


@pytest.fixture
def multi_dim_df():
    rng = np.random.default_rng(42)
    rows = []
    for period in ["2026-01", "2026-02"]:
        for region in ["华东", "华南", "华北"]:
            for channel in ["线上", "线下"]:
                base = 1000 if period == "2026-01" else 1800
                if region == "华南" and period == "2026-02":
                    base = 400  # 注入异常
                rows.append(
                    {
                        "period": period,
                        "region": region,
                        "channel": channel,
                        "sales": int(base + rng.normal(0, 50)),
                    }
                )
    return pd.DataFrame(rows)


def test_detect_anomaly_period(multi_dim_df):
    result = detect_anomaly_period(multi_dim_df, "sales", "period", threshold=0.2)
    assert result is not None
    base, current = result
    assert base == "2026-01"
    assert current == "2026-02"


def test_dimension_contribution(multi_dim_df):
    contrib = dimension_contribution(
        multi_dim_df, "region", "sales", "2026-02", "2026-01"
    )
    assert "contribution_rate" in contrib.columns
    assert contrib["contribution_rate"].sum() == pytest.approx(1.0, abs=0.01)


def test_multi_dim_drill_down(multi_dim_df):
    drill = multi_dim_drill_down(
        multi_dim_df, ["region", "channel"], "sales", "2026-02", "2026-01"
    )
    assert "level" in drill.columns
    assert len(drill) > 0


def test_additive_decomposition(multi_dim_df):
    add = additive_decomposition(
        multi_dim_df, ["region", "channel"], "sales", "2026-02", "2026-01"
    )
    assert "delta" in add.columns


def test_waterfall(multi_dim_df):
    wf = waterfall_data(multi_dim_df, "region", "sales", "2026-02", "2026-01")
    assert len(wf) >= 3
    assert wf.iloc[0]["type"] == "base"
    assert wf.iloc[-1]["type"] == "total"


def test_hypothesis_tester():
    ht = HypothesisTester(alpha=0.05)
    a = pd.Series(np.random.normal(100, 10, 50))
    b = pd.Series(np.random.normal(100, 10, 50))
    res = ht.t_test(a, b)
    assert res.p_value is not None
    assert isinstance(res.significant, bool)


def test_granger_causality():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    y = np.roll(x, 2) + rng.normal(0, 0.1, n)  # y 由 x 滞后驱动
    df = pd.DataFrame({"x": x, "y": y})
    gc = GrangerCausality(maxlag=5)
    res = gc.test(df, "y", "x")
    assert res.method == "Granger"
