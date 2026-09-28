---
name: data-intelligence-analyst
description: 数据智能分析全链路 Skill。从业务数据出发，执行数据建模、异常检测、智能归因（含因果推断）与 LLM 自动化报告。适用于数据科学/AI 分析岗的评估模型搭建、异常模式发现、指标波动根因定位与 AI Agent 分析报告生成。
---

# 数据智能分析引擎 (Data Intelligence Analyst)

面向数据科学 / AI 分析岗位的全链路分析 Skill，覆盖四大核心职责：

| 职责 | 能力 | 模块 |
|------|------|------|
| 数据建模 | 时序预测、回归评估、聚类分群 | `src/modeling/` |
| 异常检测 | 统计方法、机器学习、时序检测 | `src/anomaly/` |
| 智能归因 | 多维下钻、贡献度分解、假设检验、因果推断 | `src/attribution/` |
| AI 探索 | 可插拔 LLM 接口、自动报告、Agent 编排 | `src/llm_agent/` |

## 何时使用

当用户的需求落在以下场景时，使用本 Skill：

- 需要从业务数据中建立评估模型、预测趋势、设定目标
- 需要从海量数据中自动发现异常模式（替代人工经验规则）
- 核心指标出现波动，需要定位根因并生成分析报告
- 需要用 LLM / Agent 自动化完成数据分析与报告输出
- 需要做因果推断识别指标间因果关系（而非仅相关性）

## 数据源

本项目内置真实数据（来自火山引擎「专业数据集」vehicle_sales）：
- 比亚迪 / 特斯拉近一年全国月度销量
- 比亚迪分车系销量
- 多维增强数据（车型 × 区域 × 渠道）用于归因演示

如需接入最新真实数据，使用 `dataPro_search` 工具检索后替换 `src/utils/data_loader.py` 中的数据。

## 标准工作流

### 步骤 1：数据准备与探索

```python
import sys; sys.path.insert(0, 'src')
from utils.data_loader import prepare_all
datasets = prepare_all()
# datasets: byd_monthly, tesla_monthly, multi_dim_sales, daily_sales, ...
```

参考 Notebook：`notebooks/01_data_exploration.ipynb`

### 步骤 2：数据建模（职责1）

**时序预测** — 辅助业务目标设定：
```python
from modeling import ARIMAForecaster, compare_forecasters
series = datasets['byd_monthly'].set_index('month')['sales']
compare_forecasters(series, steps=3)  # 对比 MA / Holt-Winters / ARIMA
res = ARIMAForecaster((1,1,1)).fit_predict(series, steps=3)
```

**回归评估** — 量化影响因素：
```python
from modeling import RegressionEvaluator, compare_models
X = datasets['daily_sales'][['ad_spend', 'is_weekend', 'is_holiday']]
y = datasets['daily_sales']['sales']
compare_models(X, y)  # linear / ridge / rf / xgboost
```

**聚类分群** — 产品/区域差异化管理：
```python
from modeling import ClusterAnalyzer
sil = ClusterAnalyzer.silhouette_at_k(df, cols)  # 选 K
clustered = ClusterAnalyzer('kmeans', n_clusters=3).fit(df, cols)
```

### 步骤 3：异常检测（职责2）

**统计方法**（3σ / IQR / Z-score / ESD）：
```python
from anomaly import StatisticalAnomalyDetector
det = StatisticalAnomalyDetector('sigma', k=3).detect(series)
```

**时序检测**（滚动统计 / STL 残差 / CUSUM）：
```python
from anomaly import TimeSeriesAnomalyDetector
det = TimeSeriesAnomalyDetector('stl_residual', period=7, k=3).detect(daily_series)
```

**机器学习**（Isolation Forest / One-Class SVM / XGBoost）：
```python
from anomaly import MLAnomalyDetector
det = MLAnomalyDetector('isolation_forest', contamination=0.05).fit_predict(X)
```

### 步骤 4：智能归因（职责3）

**自动检测波动期 + 多维下钻**：
```python
from attribution import detect_anomaly_period, multi_dim_drill_down
md = datasets['multi_dim_sales'].copy()
md['period'] = md['month'].dt.strftime('%Y-%m')
base, current = detect_anomaly_period(md, 'sales', 'period', threshold=0.2)
drill = multi_dim_drill_down(md, ['region','channel','model'], 'sales', current, base)
```

**贡献度分解**：
```python
from attribution import additive_decomposition, waterfall_data
add = additive_decomposition(md, ['region','channel'], 'sales', current, base)
wf = waterfall_data(md, 'region', 'sales', current, base)  # 瀑布图
```

**假设检验验证**：
```python
from attribution import HypothesisTester
ht = HypothesisTester(alpha=0.05)
print(ht.t_test(group_a, group_b))
print(ht.mann_whitney(group_a, group_b))
```

### 步骤 5：因果推断（加分项）

**Granger 因果检验**：
```python
from attribution import GrangerCausality
gc = GrangerCausality(maxlag=7)
print(gc.test(daily, 'sales', 'ad_spend'))  # 广告是否影响销量
```

**DoWhy 因果效应估计**：
```python
from attribution import DoWhyCausalEstimator
est = DoWhyCausalEstimator('high_ad', 'sales', ['is_weekend', 'is_holiday'])
print(est.estimate(df))
```

### 步骤 6：LLM Agent 自动化报告（职责4）

**可插拔 LLM 接口**（Mock 模式无需 API Key）：
```python
from llm_agent import get_provider, ReportGenerator
llm = get_provider('mock')  # 或 'openai'（需设置 OPENAI_API_KEY）
reporter = ReportGenerator(llm)
report = reporter.generate_attribution_report(drill_df, contrib_df, hypothesis_df)
```

**端到端 Agent 编排**：
```python
from llm_agent import AnalysisAgent
agent = AnalysisAgent(df=md, metric_col='sales', period_col='period', time_col='month')
final_report = agent.run_all()  # 探索→建模→异常→归因→报告
agent.summary()  # 查看各步骤状态
```

## 可视化 Dashboard

启动 Streamlit 交互式看板：
```bash
streamlit run src/dashboard/app.py
```

## 约束与最佳实践

1. **一次只跑一个明确意图**：建模、异常、归因、因果推断分开执行，结果再汇总。
2. **归因必做显著性检验**：不要仅看贡献率就下结论，必须配合 t 检验 / Mann-Whitney。
3. **因果 ≠ 相关**：Granger 是预测因果非真实因果；DoWhy 需通过安慰剂检验。
4. **异常检测需多方法交叉验证**：统计方法 + ML + 时序检测至少两种一致才告警。
5. **LLM 报告需基于结构化结果**：先跑出数据结论，再让 LLM 生成自然语言，避免幻觉。
6. **Mock LLM 仅用于演示**：生产环境接入真实 LLM 时验证输出准确性。

## 参考 Notebooks

| Notebook | 内容 |
|----------|------|
| `01_data_exploration.ipynb` | 数据探索与可视化 |
| `02_data_modeling.ipynb` | 时序预测 / 回归 / 聚类 |
| `03_anomaly_detection.ipynb` | 统计 + ML + 时序异常检测 |
| `04_attribution_analysis.ipynb` | 多维下钻 / 贡献度 / 假设检验 |
| `05_causal_inference.ipynb` | Granger / DoWhy 因果推断 |
| `06_llm_agent_report.ipynb` | LLM 接口 / 报告 / Agent |
| `07_pro_dataset_end_to_end.ipynb` | 火山引擎专业数据集端到端 |
