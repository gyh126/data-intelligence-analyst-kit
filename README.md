# 🚗 数据智能分析引擎 (Data Intelligence Analyst Kit)

> 面向数据科学 / AI 分析岗位的全链路工具集，覆盖 **数据建模 · 异常检测 · 智能归因 · LLM Agent** 四大核心能力。

本项目旨在用数据驱动的方式辅助业务决策，支撑商务和运营决策。

## ✨ 核心能力

| JD 职责 | 模块 | 技术实现 |
|---------|------|---------|
| **1. 数据建模** | `src/modeling/` | 时序预测（ARIMA/Holt-Winters/Prophet）、回归评估（Linear/RF/XGBoost）、聚类分群 |
| **2. 异常检测** | `src/anomaly/` | 统计方法（3σ/IQR/Z-score/ESD）、机器学习（Isolation Forest/OCSVM/XGBoost）、时序检测（滚动/STL/CUSUM） |
| **3. 智能归因** | `src/attribution/` | 多维下钻、贡献度分解（加性/Shapley）、假设检验（t/Mann-Whitney/卡方/KS）、**因果推断**（Granger/DoWhy） |
| **4. AI 探索** | `src/llm_agent/` | 可插拔 LLM 接口、自动报告生成、端到端 Agent 编排 |

## 📦 项目结构

```
data-intelligence-analyst-kit/
├── src/
│   ├── modeling/          # 数据建模：时序预测 / 回归 / 聚类
│   ├── anomaly/           # 异常检测：统计 / ML / 时序
│   ├── attribution/       # 智能归因：下钻 / 贡献度 / 假设检验 / 因果推断
│   ├── llm_agent/         # LLM Agent：可插拔接口 / 报告 / 编排
│   ├── dashboard/         # Streamlit 可视化看板
│   └── utils/             # 数据加载与工具
├── notebooks/             # 7 个演示 Notebook
├── tests/                 # 单元测试（27 tests）
├── .trae/skills/          # TRAE 自定义 Skill
├── configs/               # 配置文件
├── data/                  # 数据目录
└── scripts/               # 辅助脚本
```

## 🚀 快速开始

### 安装

```bash
git clone <your-repo-url>
cd data-intelligence-analyst-kit
pip install -e ".[all]"
```

### 一键体验

```python
import sys; sys.path.insert(0, 'src')
from utils.data_loader import prepare_all
from llm_agent import AnalysisAgent

# 加载真实数据（比亚迪/特斯拉销量）
datasets = prepare_all()

# 端到端分析 Agent
md = datasets['multi_dim_sales'].copy()
md['period'] = md['month'].dt.strftime('%Y-%m')
agent = AnalysisAgent(df=md, metric_col='sales', period_col='period', time_col='month')
report = agent.run_all()
print(report)
```

### 启动可视化看板

```bash
streamlit run src/dashboard/app.py
```

## 📊 数据源

项目内置来自 **火山引擎「专业数据集」** 的真实业务数据：

- 比亚迪、特斯拉近一年全国月度销量
- 比亚迪分车系/网络销量
- 多维增强数据（车型 × 区域 × 渠道）用于归因分析演示
- 日级销量时序（含节假日/周末效应）用于异常检测

### 获取最新数据（在线模式）

项目提供 `ProDatasetClient` Python 客户端，可在 Notebook/脚本中直接调用专业数据集 API：

```python
from utils import ProDatasetClient

client = ProDatasetClient(api_key="hqd_sk_xxx")  # 或设置 PRO_DATASET_API_KEY
result = client.search("比亚迪 近一年全国销量趋势")
print(result.to_dataframe())
```

Notebook `07_pro_dataset_end_to_end.ipynb` 支持**在线/离线双模式**：
- 有 `PRO_DATASET_API_KEY` 环境变量 → 在线调用真实 API
- 无 Key → 自动回退到 `data/raw/cached_responses/` 中的缓存 JSON（已包含真实数据）

## 📚 Notebooks 演示

| Notebook | 内容 |
|----------|------|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | 数据探索与可视化 |
| [02_data_modeling.ipynb](notebooks/02_data_modeling.ipynb) | 时序预测 / 回归评估 / 聚类分群 |
| [03_anomaly_detection.ipynb](notebooks/03_anomaly_detection.ipynb) | 统计 + ML + 时序异常检测 |
| [04_attribution_analysis.ipynb](notebooks/04_attribution_analysis.ipynb) | 多维下钻 / 贡献度分解 / 假设检验 |
| [05_causal_inference.ipynb](notebooks/05_causal_inference.ipynb) | Granger / DoWhy 因果推断 |
| [06_llm_agent_report.ipynb](notebooks/06_llm_agent_report.ipynb) | LLM 接口 / 自动报告 / Agent |
| [07_pro_dataset_end_to_end.ipynb](notebooks/07_pro_dataset_end_to_end.ipynb) | 火山引擎专业数据集端到端 |

## 🧪 测试

```bash
pytest tests/ -v
```

## 🔌 LLM 接入

项目采用**可插拔 LLM 接口**设计，支持 Mock 模式（无需 API Key）和真实模型：

```python
from llm_agent import get_provider

# Mock 模式（离线演示）
llm = get_provider('mock')

# 真实 LLM（设置环境变量）
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
llm = get_provider('openai', model='gpt-4o-mini')
```

## 🤖 TRAE Skill

项目内置 TRAE 自定义 Skill：`.trae/skills/data-intelligence-analyst/SKILL.md`，封装了完整的数据分析工作流，可在 TRAE 中直接调用。

## 📄 License

MIT License

---

> 本项目为数据科学/AI 分析岗位能力展示作品集，所有真实数据来自火山引擎「专业数据集」，分析结论仅供演示。
