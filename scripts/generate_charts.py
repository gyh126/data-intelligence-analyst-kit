"""
比亚迪销量数据可视化图表生成器
输出 PNG 图表到 reports/charts/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.data_loader import load_brand_monthly, load_byd_model_aug

# 中文字体
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHART_DIR = Path(__file__).resolve().parent.parent / "reports" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {path.name}")


def chart_sales_trend():
    """图1: 比亚迪月度销量趋势 + 市占率"""
    byd = load_brand_monthly()
    byd_only = byd[byd["brand"] == "比亚迪"].copy()
    byd_only["month"] = pd.to_datetime(byd_only["month"])

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(byd_only["month"], byd_only["sales"], color="#4A90D9", alpha=0.7, label="销量(辆)")
    ax1.set_ylabel("销量(辆)", color="#4A90D9", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#4A90D9")
    ax1.set_xlabel("月份")

    ax2 = ax1.twinx()
    ax2.plot(byd_only["month"], byd_only["market_share"] * 100, "ro-", linewidth=2, markersize=6, label="市占率(%)")
    ax2.set_ylabel("市占率(%)", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")

    # 标注异常点
    anom_months = byd_only[byd_only["month"].isin(["2025-09-01", "2026-01-01", "2026-02-01"])]
    ax1.scatter(anom_months["month"], anom_months["sales"], color="orange", s=120, zorder=5, label="异常月份")

    plt.title("比亚迪月度销量趋势与市占率 (2025-09 ~ 2026-08)", fontsize=14, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95))
    fig.autofmt_xdate()
    save(fig, "01_sales_trend.png")


def chart_brand_comparison():
    """图2: 比亚迪 vs 特斯拉 月度销量对比"""
    byd = load_brand_monthly()
    byd["month"] = pd.to_datetime(byd["month"])
    byd_only = byd[byd["brand"] == "比亚迪"]
    tesla = byd[byd["brand"] == "特斯拉"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(byd_only))
    w = 0.38
    ax.bar(x - w / 2, byd_only["sales"], w, label="比亚迪", color="#4A90D9")
    ax.bar(x + w / 2, tesla["sales"], w, label="特斯拉", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(byd_only["month"].dt.strftime("%Y-%m"), rotation=45)
    ax.set_ylabel("销量(辆)")
    ax.set_title("比亚迪 vs 特斯拉 月度销量对比", fontsize=14, fontweight="bold")
    ax.legend()
    save(fig, "02_brand_comparison.png")


def chart_model_structure():
    """图3: 2026年8月车系销量结构"""
    models = load_byd_model_aug()
    top = models.nlargest(10, "sales")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top)))
    bars = ax.barh(top["model"], top["sales"], color=colors)
    ax.set_xlabel("销量(辆)")
    ax.set_title("2026年8月 比亚迪车系销量 Top 10", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, top["sales"]):
        ax.text(val + 500, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=10)
    ax.invert_yaxis()
    save(fig, "03_model_structure.png")


def chart_network_pie():
    """图4: 销售网络占比饼图"""
    models = load_byd_model_aug()
    grp = models.groupby("network")["sales"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4A90D9", "#5DADE2", "#85C1E9", "#F5B041", "#E74C3C"]
    wedges, texts, autotexts = ax.pie(
        grp.values, labels=grp.index, autopct="%1.1f%%",
        colors=colors[: len(grp)], startangle=90, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("2026年8月 各销售网络销量占比", fontsize=14, fontweight="bold")
    save(fig, "04_network_pie.png")


def chart_region_channel():
    """图5: 区域 x 渠道 热力图(8月)"""
    multi = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "multi_dim_sales.csv")
    multi["month"] = pd.to_datetime(multi["month"])
    aug = multi[multi["month"] == "2026-08-01"]
    pivot = aug.pivot_table(index="region", columns="channel", values="sales", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:,}", ha="center", va="center", fontsize=9,
                    color="white" if pivot.values[i, j] > pivot.values.max() / 2 else "black")
    plt.colorbar(im, ax=ax, label="销量(辆)")
    ax.set_title("2026年8月 区域 × 渠道 销量热力图", fontsize=14, fontweight="bold")
    save(fig, "05_region_channel_heatmap.png")


def chart_anomaly():
    """图6: 异常检测可视化"""
    byd = load_brand_monthly()
    byd_only = byd[byd["brand"] == "比亚迪"].copy()
    byd_only["month"] = pd.to_datetime(byd_only["month"])
    s = byd_only.set_index("month")["sales"]

    mean = s.mean()
    std = s.std()
    upper = mean + 1.5 * std
    lower = mean - 1.5 * std

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index, s.values, "b-o", linewidth=2, label="实际销量")
    ax.axhline(upper, color="red", linestyle="--", label=f"上界 (μ+1.5σ={upper:,.0f})")
    ax.axhline(lower, color="red", linestyle="--", label=f"下界 (μ-1.5σ={lower:,.0f})")
    ax.axhline(mean, color="gray", linestyle=":", label=f"均值={mean:,.0f}")

    anom_mask = (s > upper) | (s < lower)
    ax.scatter(s.index[anom_mask], s.values[anom_mask], color="red", s=150, zorder=5, label="异常点")
    for idx in s.index[anom_mask]:
        ax.annotate(idx.strftime("%Y-%m"), (idx, s[idx]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color="red", fontweight="bold")

    ax.set_ylabel("销量(辆)")
    ax.set_title("异常检测结果 (Z-score, 阈值=1.5σ)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    save(fig, "06_anomaly_detection.png")


def chart_forecast():
    """图7: 销量预测"""
    from modeling.time_series import ExponentialSmoothingForecaster

    byd = load_brand_monthly()
    byd_only = byd[byd["brand"] == "比亚迪"].copy()
    byd_only["month"] = pd.to_datetime(byd_only["month"])
    s = byd_only.set_index("month")["sales"]

    forecaster = ExponentialSmoothingForecaster(trend="add")
    result = forecaster.fit_predict(s, steps=3)
    fc = result.forecast

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.index, s.values, "b-o", linewidth=2, label="历史销量")
    ax.plot(fc.index, fc.values, "r--s", linewidth=2, markersize=8, label="预测销量")
    ax.axvline(s.index[-1], color="gray", linestyle=":", alpha=0.5)
    ax.text(s.index[-1], ax.get_ylim()[1] * 0.95, "预测起点", ha="right", fontsize=10, color="gray")

    for idx, val in fc.items():
        ax.annotate(f"{val:,.0f}", (idx, val), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, color="red")

    ax.set_ylabel("销量(辆)")
    ax.set_title("比亚迪销量预测 (Holt-Winters, 未来3月)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.autofmt_xdate()
    save(fig, "07_forecast.png")


if __name__ == "__main__":
    print("生成可视化图表...")
    chart_sales_trend()
    chart_brand_comparison()
    chart_model_structure()
    chart_network_pie()
    chart_region_channel()
    chart_anomaly()
    chart_forecast()
    print(f"\n✅ 全部图表已保存到 {CHART_DIR}")
