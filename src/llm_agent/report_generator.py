"""
报告生成器
职责3+4-智能归因 × AI 探索：将分析结果转化为自然语言报告。
支持：结构化数据摘要、归因报告、异常报告、综合分析报告。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .provider import BaseLLMProvider, MockLLMProvider


class ReportGenerator:
    """基于 LLM 的分析报告生成器"""

    def __init__(self, llm: Optional[BaseLLMProvider] = None):
        self.llm = llm or MockLLMProvider()

    def _data_summary(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """生成数据摘要文本"""
        lines = [
            f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列",
            f"字段：{', '.join(df.columns.tolist())}",
            "",
        ]
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines.append("数值字段统计：")
            desc = numeric.describe().round(2)
            lines.append(desc.to_string())
        lines.append("")
        lines.append(f"前 {max_rows} 行：")
        lines.append(df.head(max_rows).to_string())
        return "\n".join(lines)

    def generate_exploration_report(self, df: pd.DataFrame) -> str:
        """生成探索性分析报告"""
        summary = self._data_summary(df)
        prompt = f"""请基于以下数据概况，生成一份探索性数据分析报告，
包括数据质量、分布特征、趋势与相关性洞察。

{summary}
"""
        resp = self.llm.chat(prompt, system="你是一位资深数据分析师。")
        return resp.content

    def generate_anomaly_report(
        self,
        series: pd.Series,
        anomaly_df: pd.DataFrame,
        method: str = "混合检测",
    ) -> str:
        """生成异常检测报告"""
        anom_count = int(anomaly_df["is_anomaly"].sum()) if "is_anomaly" in anomaly_df else 0
        total = len(anomaly_df)
        prompt = f"""请生成异常检测分析报告。

检测方法：{method}
检测数据点数：{total}
检出异常数：{anom_count}（占比 {anom_count/total*100:.1f}%）
异常点详情：
{anomaly_df[anomaly_df.get('is_anomaly', pd.Series(False))].head(10).to_string()}

请分析异常模式、可能原因，并给出将人工规则升级为算法方案的建议。
"""
        resp = self.llm.chat(prompt, system="你是一位擅长时序异常检测的数据科学家。")
        return resp.content

    def generate_attribution_report(
        self,
        drill_down_df: pd.DataFrame,
        contribution_df: pd.DataFrame,
        hypothesis_result: Optional[pd.DataFrame] = None,
    ) -> str:
        """生成智能归因报告"""
        prompt = f"""请生成智能归因分析报告。

多维下钻结果：
{drill_down_df.head(10).to_string()}

贡献度分解（Top 10）：
{contribution_df.head(10).to_string()}

假设检验结果：
{hypothesis_result.to_string() if hypothesis_result is not None else '未执行'}

请分析核心指标波动的根因，给出统计显著性结论与业务建议。
"""
        resp = self.llm.chat(prompt, system="你是一位擅长业务归因分析的高级数据分析师。")
        return resp.content

    def generate_forecast_report(
        self,
        history: pd.Series,
        forecast: pd.Series,
        metrics: dict,
    ) -> str:
        """生成预测报告"""
        forecast_text = "\n".join(
            [f"- {idx.strftime('%Y-%m')}: {val:,.0f}" for idx, val in forecast.items()]
        )
        prompt = f"""请生成销量预测与目标设定报告。

模型评估指标：{metrics}
历史销量趋势：共 {len(history)} 期
未来预测：
{forecast_text}

请基于预测结果给出业务目标设定建议。
"""
        resp = self.llm.chat(prompt, system="你是一位擅长业务预测与目标管理的数据分析师。")
        return resp.content
