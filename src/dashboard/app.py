"""
Streamlit Dashboard — 数据智能分析引擎可视化界面
职责3-智能归因 × Streamlit：交互式展示归因分析与异常检测结果。

启动：streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 确保能导入项目模块
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.data_loader import prepare_all  # noqa: E402
from anomaly import StatisticalAnomalyDetector, TimeSeriesAnomalyDetector  # noqa: E402
from attribution import (  # noqa: E402
    additive_decomposition,
    dimension_contribution,
    waterfall_data,
)
from modeling import ARIMAForecaster, compare_forecasters  # noqa: E402

st.set_page_config(page_title="数据智能分析引擎", layout="wide")


@st.cache_data
def load_data():
    return prepare_all()


def main():
    st.title("🚗 数据智能分析引擎")
    st.caption("数据建模 · 异常检测 · 智能归因 · LLM Agent — 基于比亚迪真实销量数据")

    datasets = load_data()
    byd = datasets["byd_monthly"]
    md = datasets["multi_dim_sales"].copy()
    md["period"] = md["month"].dt.strftime("%Y-%m")
    daily = datasets["daily_sales"]

    # 侧边栏
    st.sidebar.header("分析模块")
    module = st.sidebar.radio(
        "选择模块",
        ["📈 销量总览", "🔮 数据建模", "⚠️ 异常检测", "🎯 智能归因", "🤖 LLM 报告"],
    )

    if module == "📈 销量总览":
        show_overview(byd, md)
    elif module == "🔮 数据建模":
        show_modeling(byd)
    elif module == "⚠️ 异常检测":
        show_anomaly(byd, daily)
    elif module == "🎯 智能归因":
        show_attribution(md)
    elif module == "🤖 LLM 报告":
        show_llm_report(md, byd)


def show_overview(byd: pd.DataFrame, md: pd.DataFrame):
    st.header("📈 销量总览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总销量", f"{byd['sales'].sum():,}", "辆")
    with col2:
        st.metric("平均月度销量", f"{byd['sales'].mean():,.0f}", "辆")
    with col3:
        st.metric("最高市场份额", f"{byd['market_share'].max()*100:.1f}%")

    # 月度趋势
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=byd["month"],
            y=byd["sales"],
            mode="lines+markers",
            name="销量",
            line=dict(width=3),
        )
    )
    fig.update_layout(title="比亚迪月度销量趋势", xaxis_title="月份", yaxis_title="销量")
    st.plotly_chart(fig, use_container_width=True)

    # 车型分布
    st.subheader("车型销量分布")
    model_sales = md.groupby("model")["sales"].sum().sort_values(ascending=False)
    fig2 = px.bar(
        x=model_sales.index,
        y=model_sales.values,
        title="各车型累计销量",
        labels={"x": "车型", "y": "销量"},
    )
    st.plotly_chart(fig2, use_container_width=True)


def show_modeling(byd: pd.DataFrame):
    st.header("🔮 数据建模 — 销量预测")
    series = byd.set_index("month")["sales"]

    steps = st.slider("预测期数", 1, 6, 3)
    model_choice = st.selectbox(
        "预测模型", ["ARIMA(1,1,1)", "移动平均", "Holt-Winters"]
    )

    from modeling import (
        ARIMAForecaster,
        ExponentialSmoothingForecaster,
        MovingAverageForecaster,
    )

    if model_choice == "ARIMA(1,1,1)":
        res = ARIMAForecaster((1, 1, 1)).fit_predict(series, steps)
    elif model_choice == "移动平均":
        res = MovingAverageForecaster(3).fit_predict(series, steps)
    else:
        res = ExponentialSmoothingForecaster().fit_predict(series, steps)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAPE", f"{res.metrics.get('MAPE(%)', 'N/A')}%")
    with col2:
        st.metric("RMSE", f"{res.metrics.get('RMSE', 'N/A')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="历史", line=dict(width=3)))
    fig.add_trace(
        go.Scatter(
            x=res.forecast.index,
            y=res.forecast.values,
            name="预测",
            line=dict(dash="dash", width=3),
        )
    )
    fig.update_layout(title=f"销量预测 ({res.model_name})")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("模型对比")
    st.dataframe(compare_forecasters(series, steps))


def show_anomaly(byd: pd.DataFrame, daily: pd.DataFrame):
    st.header("⚠️ 异常检测")

    tab1, tab2 = st.tabs(["月度销量异常", "日级销量异常"])

    with tab1:
        series = byd.set_index("month")["sales"]
        method = st.selectbox(
            "检测方法",
            ["3σ 准则", "IQR", "Z-score", "滚动统计"],
            key="m_anom",
        )
        if method == "3σ 准则":
            det = StatisticalAnomalyDetector("sigma", k=2).detect(series)
        elif method == "IQR":
            det = StatisticalAnomalyDetector("iqr").detect(series)
        elif method == "Z-score":
            det = StatisticalAnomalyDetector("zscore", threshold=1.5).detect(series)
        else:
            det = TimeSeriesAnomalyDetector("rolling", window=3, k=2).detect(series)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=det.index, y=det["value"], name="销量", line=dict(width=3)))
        if "upper" in det.columns:
            fig.add_trace(
                go.Scatter(
                    x=det.index, y=det["upper"], name="上界", line=dict(dash="dot")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=det.index, y=det["lower"], name="下界", line=dict(dash="dot")
                )
            )
        anom = det[det["is_anomaly"]]
        if len(anom) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anom.index,
                    y=anom["value"],
                    mode="markers",
                    name="异常点",
                    marker=dict(size=12, color="red"),
                )
            )
        fig.update_layout(title=f"异常检测 ({method})")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"检出 {int(det['is_anomaly'].sum())} 个异常点")

    with tab2:
        daily_series = daily.set_index("date")["sales"]
        det2 = TimeSeriesAnomalyDetector("stl_residual", period=7, k=3).detect(daily_series)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=det2.index, y=det2["value"], name="销量", line=dict(width=1)))
        anom2 = det2[det2["is_anomaly"]]
        fig2.add_trace(
            go.Scatter(
                x=anom2.index,
                y=anom2["value"],
                mode="markers",
                name="异常点",
                marker=dict(size=8, color="red"),
            )
        )
        fig2.update_layout(title="日级销量 STL 残差异常检测")
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"检出 {int(det2['is_anomaly'].sum())} 个异常点")


def show_attribution(md: pd.DataFrame):
    st.header("🎯 智能归因分析")

    periods = sorted(md["period"].unique())
    base, current = st.select_slider(
        "选择对比期间",
        options=periods,
        value=(periods[-2], periods[-1]),
    )

    dim = st.selectbox("归因维度", ["region", "model", "channel"])

    contrib = dimension_contribution(md, dim, "sales", current, base)
    st.subheader(f"{dim} 维度贡献度")
    st.dataframe(contrib, use_container_width=True)

    # 瀑布图
    wf = waterfall_data(md, dim, "sales", current, base)
    fig = go.Figure(
        go.Waterfall(
            name="销量变动",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(wf) - 2) + ["total"],
            x=wf["item"],
            y=wf["delta"].where(wf["type"] != "base", wf["value"]),
            text=wf["value"].round(0),
        )
    )
    fig.update_layout(title=f"销量变动瀑布图 ({base} → {current})")
    st.plotly_chart(fig, use_container_width=True)

    # 多维下钻
    st.subheader("多维下钻")
    from attribution import multi_dim_drill_down

    drill = multi_dim_drill_down(
        md, ["region", "channel", "model"], "sales", current, base, top_n=3
    )
    st.dataframe(drill, use_container_width=True)


def show_llm_report(md: pd.DataFrame, byd: pd.DataFrame):
    st.header("🤖 LLM 智能分析报告")
    st.caption("可插拔 LLM 接口 — 当前使用 Mock/Demo 模式（无需 API Key）")

    report_type = st.selectbox(
        "报告类型",
        ["探索性分析", "异常检测报告", "归因分析报告", "预测报告"],
    )

    from llm_agent import MockLLMProvider, ReportGenerator

    reporter = ReportGenerator(MockLLMProvider())

    if report_type == "探索性分析":
        report = reporter.generate_exploration_report(md)
    elif report_type == "异常检测报告":
        series = byd.set_index("month")["sales"]
        det = StatisticalAnomalyDetector("sigma", k=2).detect(series)
        report = reporter.generate_anomaly_report(series, det)
    elif report_type == "归因分析报告":
        periods = sorted(md["period"].unique())
        from attribution import additive_decomposition, dimension_contribution

        contrib = dimension_contribution(
            md, "region", "sales", periods[-1], periods[-2]
        )
        add = additive_decomposition(md, ["region"], "sales", periods[-1], periods[-2])
        report = reporter.generate_attribution_report(contrib, add)
    else:
        series = byd.set_index("month")["sales"]
        res = ARIMAForecaster((1, 1, 1)).fit_predict(series, steps=3)
        report = reporter.generate_forecast_report(series, res.forecast, res.metrics)

    st.markdown(report)


if __name__ == "__main__":
    main()
