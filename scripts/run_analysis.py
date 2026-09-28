"""
通用数据分析 CLI
支持任意数据源（专业数据集 API 或本地 CSV），自动完成探索/异常/归因/预测，输出通用报告+图表。

用法:
  # 1. 从专业数据集检索数据并分析
  python scripts/run_analysis.py --query "贵州茅台 近一年股价" --metric 收盘价 --time 日期

  # 2. 从本地 CSV 分析
  python scripts/run_analysis.py --csv data/processed/byd_monthly.csv --metric sales --time month

  # 3. 指定维度列做归因
  python scripts/run_analysis.py --query "比亚迪 分车系销量" --metric 销量 --time 月份 --dims 车系,网络
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.pro_dataset_client import ProDatasetClient

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fetch_data(args):
    """获取数据：优先 CSV，其次专业数据集 API"""
    if args.csv:
        print(f"📂 从本地 CSV 加载: {args.csv}")
        df = pd.read_csv(args.csv)
    elif args.query:
        api_key = args.api_key or __import__("os").getenv("PRO_DATASET_API_KEY")
        if not api_key:
            print("❌ 需要 API Key: 用 --api-key 或设置 PRO_DATASET_API_KEY")
            sys.exit(1)
        print(f"🔍 检索专业数据集: {args.query}")
        client = ProDatasetClient(api_key=api_key)
        df = client.search_to_dataframe(args.query)
        print(f"   获取 {len(df)} 行 × {len(df.columns)} 列")
    else:
        print("❌ 请指定 --csv 或 --query")
        sys.exit(1)

    # 尝试将看起来像日期的列转为 datetime
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() / len(df) > 0.8:
                    df[col] = converted
            except Exception:
                pass
    return df


def auto_detect(df, args):
    """自动推断指标列、时间列、维度列"""
    metric = args.metric
    time_col = args.time
    dims = args.dims.split(",") if args.dims else None

    if metric is None:
        # 选第一个数值列
        numeric = df.select_dtypes(include=[np.number]).columns
        metric = numeric[0] if len(numeric) else df.columns[-1]
        print(f"🔧 自动推断指标列: {metric}")

    if time_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ["date", "month", "time", "日期", "月份", "时间"]):
                time_col = c
                break
        if time_col is None:
            time_col = df.columns[0]
        print(f"🔧 自动推断时间列: {time_col}")

    if dims is None:
        dims = [c for c in df.columns if c not in [metric, time_col] and df[c].dtype == object and df[c].nunique() < 50]
        print(f"🔧 自动推断维度列: {dims}")

    return metric, time_col, dims


def gen_trend_chart(df, metric, time_col, out_path):
    """趋势图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = df.groupby(time_col)[metric].sum()
    ax.plot(grouped.index, grouped.values, "b-o", linewidth=2)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} 趋势", fontsize=14, fontweight="bold")
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gen_anomaly_chart(df, metric, time_col, out_path):
    """异常检测图"""
    from anomaly.statistical import StatisticalAnomalyDetector
    s = df.groupby(time_col)[metric].sum()
    det = StatisticalAnomalyDetector(method="zscore", threshold=2.0)
    res = det.detect(s)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s.index, s.values, "b-o", linewidth=2, label="实际值")
    if "upper" in res.columns:
        ax.axhline(res["upper"].iloc[0], color="red", ls="--", label="上界")
        ax.axhline(res["lower"].iloc[0], color="red", ls="--", label="下界")
    anom = res[res["is_anomaly"]]
    if not anom.empty:
        ax.scatter(anom.index, anom["value"], color="red", s=120, zorder=5, label="异常点")
    ax.set_ylabel(metric)
    ax.set_title(f"异常检测 (Z-score, 阈值=2σ)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(anom)


def gen_dim_chart(df, metric, dims, out_path):
    """维度占比图（取第一个维度）"""
    if not dims:
        return
    dim = dims[0]
    grp = df.groupby(dim)[metric].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grp.index, grp.values, color="steelblue")
    ax.set_ylabel(metric)
    ax.set_title(f"按 {dim} 的 {metric} 分布 (Top 10)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gen_forecast_chart(df, metric, time_col, out_path):
    """预测图"""
    from modeling.time_series import ExponentialSmoothingForecaster
    s = df.groupby(time_col)[metric].sum()
    try:
        fc = ExponentialSmoothingForecaster(trend="add").fit_predict(s, steps=3).forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(s.index, s.values, "b-o", linewidth=2, label="历史")
        ax.plot(fc.index, fc.values, "r--s", linewidth=2, label="预测")
        ax.axvline(s.index[-1], color="gray", ls=":", alpha=0.5)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} 预测 (未来3期)", fontsize=14, fontweight="bold")
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return fc
    except Exception as e:
        print(f"   ⚠️ 预测跳过: {e}")
        return None


def build_report(df, metric, time_col, dims, anom_count, forecast, out_dir):
    """生成 Markdown 报告"""
    desc = df[metric].describe()
    md = f"""# 数据分析报告

> 数据源：{'专业数据集' if 'query' in str(out_dir) else '本地CSV'} · 指标：{metric} · 时间列：{time_col}

## 数据概览
- 样本量：{len(df):,} 行
- 字段数：{len(df.columns)} 列
- 指标列：{metric}（均值 {desc['mean']:,.2f}，最大 {desc['max']:,.2f}，最小 {desc['min']:,.2f}）

## 趋势分析
![趋势](trend.png)

## 异常检测
- 检出异常点数：**{anom_count}** 个（Z-score 阈值=2σ）
![异常检测](anomaly.png)

## 维度分析
{dims and f"分析维度：{', '.join(dims)}" or "无维度列"}
![维度分布](dimension.png)

## 预测
{forecast is not None and "已生成未来3期预测" or "数据不足，跳过预测"}
![预测](forecast.png)

## 结论
1. {metric} 的整体趋势：均值 {desc['mean']:,.2f}，标准差 {desc['std']:,.2f}。
2. 异常检测共发现 {anom_count} 个异常点，建议结合业务上下文核查。
3. 预测结果可作为目标设定参考。

---
*由 data-intelligence-analyst-kit 通用分析引擎自动生成*
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"✅ 报告: {out_dir / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description="通用数据分析引擎")
    parser.add_argument("--query", help="专业数据集查询语句（如 '贵州茅台 近一年股价'）")
    parser.add_argument("--csv", help="本地 CSV 文件路径")
    parser.add_argument("--api-key", help="专业数据集 API Key（或用 PRO_DATASET_API_KEY 环境变量）")
    parser.add_argument("--metric", help="指标列名（不填则自动推断第一个数值列）")
    parser.add_argument("--time", help="时间列名（不填则自动推断）")
    parser.add_argument("--dims", help="维度列名，逗号分隔（不填则自动推断）")
    parser.add_argument("--output", default="analysis_output", help="输出目录")
    args = parser.parse_args()

    # 1. 获取数据
    df = fetch_data(args)
    if df.empty:
        print("❌ 数据为空")
        sys.exit(1)

    # 2. 自动推断列
    metric, time_col, dims = auto_detect(df, args)
    print(f"\n📊 分析配置: 指标={metric}, 时间={time_col}, 维度={dims}")

    # 3. 输出目录
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 生成图表
    print("\n生成图表...")
    gen_trend_chart(df, metric, time_col, out_dir / "trend.png")
    print("  ✅ trend.png")
    anom_count = gen_anomaly_chart(df, metric, time_col, out_dir / "anomaly.png")
    print(f"  ✅ anomaly.png (异常点: {anom_count})")
    if dims:
        gen_dim_chart(df, metric, dims, out_dir / "dimension.png")
        print("  ✅ dimension.png")
    else:
        # 生成空图占位
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, "无维度列", ha="center", va="center", fontsize=16)
        plt.savefig(out_dir / "dimension.png", dpi=120, bbox_inches="tight")
        plt.close()
    forecast = gen_forecast_chart(df, metric, time_col, out_dir / "forecast.png")
    print("  ✅ forecast.png")

    # 5. 生成报告
    build_report(df, metric, time_col, dims, anom_count, forecast, out_dir)
    print(f"\n🎉 分析完成！输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
