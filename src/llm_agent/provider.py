"""
LLM Provider 可插拔接口
职责4-AI 探索：统一 LLM 调用接口，支持 Demo/Mock 模式，预留真实 API 接入。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[dict] = None
    raw: Optional[object] = None


class BaseLLMProvider(ABC):
    """LLM 提供者抽象基类"""

    name: str = "base"

    @abstractmethod
    def chat(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """单次对话"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """是否可用"""
        ...


class MockLLMProvider(BaseLLMProvider):
    """
    Demo/Mock LLM：无需 API Key，基于规则模板生成结构化分析文本。
    用于离线演示与开发调试。
    """

    name = "mock"

    def __init__(self):
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def chat(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        self.call_count += 1
        # 根据 prompt 关键词生成对应模板
        content = self._template_response(prompt)
        return LLMResponse(
            content=content,
            model="mock-llm-v1",
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(content)},
        )

    def _template_response(self, prompt: str) -> str:
        """根据分析类型生成模板化报告"""
        p = prompt.lower()
        if "异常" in prompt or "anomaly" in p:
            return self._anomaly_report()
        if "归因" in prompt or "attribution" in p or "波动" in prompt:
            return self._attribution_report()
        if "预测" in prompt or "forecast" in p:
            return self._forecast_report()
        if "建模" in prompt or "model" in p:
            return self._modeling_report()
        return self._general_report(prompt)

    def _anomaly_report(self) -> str:
        return """## 异常检测分析报告

**检测结论**：在观测周期内共识别出 3 个显著异常点。

**异常详情**：
1. 2026-02 华南经销商渠道销量同比下滑 65%（触发 3σ + IQR 双重规则）
   - 疑似原因：春节假期叠加区域库存调整
2. 2026-06 海豹06 全渠道销量环比下降 60%
   - 疑似原因：车型改款切换期，老款清库与新款上市空档
3. 2026-01 整体销量环比下降 74%
   - 疑似原因：春节前置效应

**算法方案建议**：
- 将人工规则（春节-30%、车型切换-50%）升级为时序异常检测算法
- 采用 Isolation Forest 多维检测 + STL 残差 3σ 双重校验
- 异常告警阈值建议设为 contamination=0.05
"""

    def _attribution_report(self) -> str:
        return """## 智能归因分析报告

**核心问题**：2026年2月销量环比下滑的根因定位

**多维下钻路径**：
区域(华南, 贡献 -38%) → 渠道(经销商, 贡献 -52%) → 车型(海豹06, 贡献 -27%)

**贡献度分解（Top 3）**：
| 维度组合 | 贡献率 | 方向 |
|---------|--------|------|
| 华南 × 经销商 × 海豹06 | -32% | 负向 |
| 华东 × 经销商 × 秦PLUS | -18% | 负向 |
| 华北 × 线上直营 × 海豚 | +12% | 正向 |

**统计验证**：
- 华南经销商 2月 vs 1月：t 检验 p=0.003（显著）
- 海豹06 2月 vs 上月：Mann-Whitney p=0.018（显著）

**根因结论**：
销量下滑主要由「华南区域经销商渠道」在「海豹06车型」上的异常下跌驱动，
具有统计显著性。建议排查该区域经销商库存与促销政策。
"""

    def _forecast_report(self) -> str:
        return """## 销量预测与目标设定报告

**预测方法对比**：
| 模型 | MAPE | RMSE | 适用场景 |
|------|------|------|---------|
| ARIMA(1,1,1) | 8.2% | 12,450 | 短中期趋势 |
| Holt-Winters | 6.5% | 9,820 | 含季节性 |
| Prophet | 5.1% | 7,630 | 节假日效应 |

**未来3月预测**（Prophet）：
- 2026-09：195,000 辆（区间 [180k, 210k]）
- 2026-10：210,000 辆（区间 [195k, 225k]）
- 2026-11：205,000 辆（区间 [190k, 220k]）

**目标建议**：
四季度目标设定为 610,000 辆，基于预测中位数上浮 3% 作为进取目标。
需重点关注华南经销商渠道恢复情况。
"""

    def _modeling_report(self) -> str:
        return """## 业务评估建模报告

**评估模型**：XGBoost 回归（销量预测）

**模型表现**：
- R² = 0.87
- RMSE = 8,420
- MAE = 6,150

**特征重要性 Top 5**：
1. 广告投放费用（贡献度 0.32）
2. 是否周末（贡献度 0.21）
3. 是否节假日（贡献度 0.18）
4. 时间趋势（贡献度 0.15）
5. 区域 × 渠道交互（贡献度 0.14）

**业务洞察**：
广告投放对销量有显著正向影响，滞后 2-3 天效果最佳。
建议在周末与节假日前加大投放力度。
"""

    def _general_report(self, prompt: str) -> str:
        return f"""## 数据分析报告

> 本报告由 Mock LLM 生成（Demo 模式）。接入真实 LLM 后将生成更具针对性的分析。

**分析主题**：{prompt[:100]}

**分析框架**：
1. 数据概况与质量评估
2. 探索性分析（趋势/分布/相关性）
3. 建模与预测
4. 异常检测与归因
5. 结论与建议

**建议**：请提供具体的分析目标与数据上下文，以获得更精准的分析结论。
"""


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / 兼容接口 LLM 提供者"""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请先安装 openai: pip install openai") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=self.model, messages=messages)
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        )


def get_provider(name: str = "mock", **kwargs) -> BaseLLMProvider:
    """工厂方法：获取 LLM 提供者"""
    if name == "mock":
        return MockLLMProvider()
    if name in ("openai", "gpt"):
        return OpenAIProvider(**kwargs)
    raise ValueError(f"未知 LLM 提供者: {name}")
