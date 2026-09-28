"""数据层测试"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.data_loader import (  # noqa: E402
    generate_daily_sales,
    generate_multi_dim_sales,
    load_byd_monthly,
    load_brand_monthly,
    load_tesla_monthly,
    prepare_all,
)


def test_load_byd_monthly():
    df = load_byd_monthly()
    assert len(df) == 12
    assert "sales" in df.columns
    assert df["sales"].sum() > 0


def test_load_brand_monthly():
    df = load_brand_monthly()
    assert "比亚迪" in df["brand"].values
    assert "特斯拉" in df["brand"].values


def test_generate_multi_dim_sales():
    df = generate_multi_dim_sales()
    assert "model" in df.columns
    assert "region" in df.columns
    assert "channel" in df.columns
    assert df["sales"].dtype in (np.int64, np.int32, int)


def test_generate_daily_sales():
    df = generate_daily_sales(days=100)
    assert len(df) == 100
    assert "anomaly_true" in df.columns
    assert df["anomaly_true"].sum() > 0  # 包含注入异常


def test_prepare_all():
    ds = prepare_all()
    assert "byd_monthly" in ds
    assert "multi_dim_sales" in ds
    assert "daily_sales" in ds
