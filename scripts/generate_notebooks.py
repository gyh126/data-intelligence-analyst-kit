"""
批量生成演示 Notebook 的脚本。
运行：python scripts/generate_notebooks.py
"""
import json
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def save_nb(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✓ {name}")


# ============================================================
# 01. 数据探索
# ============================================================
save_nb(
    "01_data_exploration.ipynb",
    [
        md("# 01. 数据探索 (EDA)\n\n"
           "职责1-数据建模：从原始数据中独立完成探索性分析。\n\n"
           "**数据源**：火山引擎「专业数据集」vehicle_sales（比亚迪/特斯拉真实销量）+ 多维增强数据。"),
        md("## 1. 环境与数据加载"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import pandas as pd\n"
             "import matplotlib.pyplot as plt\n"
             "import seaborn as sns\n"
             "sns.set_style('whitegrid')\n"
             "%matplotlib inline\n\n"
             "from utils.data_loader import prepare_all\n"
             "ds = prepare_all()\n"
             "list(ds.keys())"),
        md("## 2. 真实数据概况"),
        code("byd = ds['byd_monthly']\n"
             "print('比亚迪月度销量:')\n"
             "display(byd)\n"
             "print('\\n特斯拉月度销量:')\n"
             "display(ds['tesla_monthly'])"),
        md("## 3. 销量趋势可视化"),
        code("fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n"
             "byd.plot(x='month', y='sales', ax=axes[0], title='比亚迪月度销量', marker='o')\n"
             "ds['tesla_monthly'].plot(x='month', y='sales', ax=axes[1], title='特斯拉月度销量', marker='o', color='orange')\n"
             "plt.tight_layout()"),
        md("## 4. 多维度数据分布"),
        code("md = ds['multi_dim_sales']\n"
             "print('多维数据 shape:', md.shape)\n"
             "display(md.head())\n\n"
             "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
             "md.groupby('region')['sales'].sum().plot(kind='bar', ax=axes[0], title='区域销量分布')\n"
             "md.groupby('channel')['sales'].sum().plot(kind='bar', ax=axes[1], title='渠道销量分布')\n"
             "md.groupby('model')['sales'].sum().sort_values().plot(kind='barh', ax=axes[2], title='车型销量分布')\n"
             "plt.tight_layout()"),
        md("## 5. 相关性分析"),
        code("pivot = md.pivot_table(index='month', columns='region', values='sales', aggfunc='sum')\n"
             "sns.heatmap(pivot.corr(), annot=True, cmap='coolwarm', fmt='.2f')\n"
             "plt.title('区域间销量相关性')"),
        md("## 6. 日级数据探索"),
        code("daily = ds['daily_sales']\n"
             "print(daily.shape)\n"
             "display(daily.describe())\n"
             "daily.plot(x='date', y='sales', figsize=(14, 4), title='日级销量时序')"),
        md("## 小结\n\n"
           "- 真实数据覆盖比亚迪/特斯拉近12个月销量\n"
           "- 多维数据含 车型×区域×渠道 三维结构，适合归因分析\n"
           "- 日级数据含节假日/周末效应，适合时序异常检测"),
    ],
)


# ============================================================
# 02. 数据建模
# ============================================================
save_nb(
    "02_data_modeling.ipynb",
    [
        md("# 02. 数据建模\n\n"
           "职责1-数据建模：建立评估模型，指导业务制定目标。\n\n"
           "涵盖：时序预测（ARIMA/Holt-Winters/Prophet）、回归评估、聚类分群。"),
        md("## 1. 时序预测 — 销量预测与目标设定"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import pandas as pd\n"
             "from utils.data_loader import load_byd_monthly\n"
             "from modeling import (ARIMAForecaster, ExponentialSmoothingForecaster, \n"
             "                      MovingAverageForecaster, compare_forecasters)\n\n"
             "series = load_byd_monthly().set_index('month')['sales']\n"
             "series"),
        code("# 模型对比\n"
             "compare_forecasters(series, steps=3)"),
        code("# ARIMA 预测\n"
             "res = ARIMAForecaster((1,1,1)).fit_predict(series, steps=3)\n"
             "print('预测指标:', res.metrics)\n"
             "print('未来3月预测:')\n"
             "print(res.forecast)"),
        md("## 2. 回归评估模型 — 量化影响因素"),
        code("from modeling import RegressionEvaluator, compare_models\n\n"
             "daily = ds['daily_sales'] if 'ds' in dir() else __import__('utils.data_loader', fromlist=['prepare_all']).prepare_all()['daily_sales']\n"
             "X = daily[['ad_spend', 'is_weekend', 'is_holiday']]\n"
             "y = daily['sales']\n"
             "compare_models(X, y)"),
        code("# XGBoost 特征重要性\n"
             "res = RegressionEvaluator('xgboost').fit(X, y)\n"
             "print(res.summary())\n"
             "res.feature_importance.plot(kind='barh', title='特征重要性')"),
        md("## 3. 聚类分析 — 产品/区域分群"),
        code("from modeling import ClusterAnalyzer\n"
             "md = ds['multi_dim_sales']\n"
             "profile = md.groupby('model').agg({'sales': ['sum', 'mean', 'std']}).round(0)\n"
             "profile.columns = ['total', 'mean', 'std']\n"
             "profile = profile.fillna(0)\n\n"
             "# 选 K\n"
             "sil = ClusterAnalyzer.silhouette_at_k(profile.reset_index(), ['total','mean','std'])\n"
             "print(sil)\n\n"
             "# K=3 聚类\n"
             "clustered = ClusterAnalyzer('kmeans', n_clusters=3).fit(profile.reset_index(), ['total','mean','std'])\n"
             "print(clustered[['model','cluster']])"),
        md("## 小结\n\n"
           "- 时序预测辅助业务设定月度/季度目标\n"
           "- 回归模型量化广告投放等因素的贡献\n"
           "- 聚类识别产品线结构，支持差异化目标管理"),
    ],
)


# ============================================================
# 03. 异常检测
# ============================================================
save_nb(
    "03_anomaly_detection.ipynb",
    [
        md("# 03. 异常检测\n\n"
           "职责2-异常检测：统计模型 + 机器学习，从数据中自动发现异常模式，将人工经验规则升级为算法方案。\n\n"
           "涵盖：3σ/IQR/Z-score/ESD 统计方法、Isolation Forest/One-Class SVM/XGBoost、滚动统计/STL/CUSUM 时序检测。"),
        md("## 1. 统计方法对比"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import pandas as pd\n"
             "from utils.data_loader import load_byd_monthly\n"
             "from anomaly import StatisticalAnomalyDetector\n\n"
             "series = load_byd_monthly().set_index('month')['sales']\n"
             "for method in ['sigma', 'iqr', 'zscore', 'esd']:\n"
             "    det = StatisticalAnomalyDetector(method).detect(series)\n"
             "    print(f'{method:8s}: {int(det[\"is_anomaly\"].sum())} anomalies')"),
        md("## 2. 时序异常检测"),
        code("from anomaly import TimeSeriesAnomalyDetector\n"
             "from utils.data_loader import prepare_all\n"
             "daily = prepare_all()['daily_sales'].set_index('date')['sales']\n\n"
             "# STL 残差异常\n"
             "det = TimeSeriesAnomalyDetector('stl_residual', period=7, k=3).detect(daily)\n"
             "print(f'STL 残差: {int(det[\"is_anomaly\"].sum())} anomalies')\n\n"
             "# CUSUM 均值漂移\n"
             "cusum = TimeSeriesAnomalyDetector('cusum', target=daily.mean(), std=daily.std(), k=0.5, h=5).detect(daily)\n"
             "print(f'CUSUM: {int(cusum[\"is_anomaly\"].sum())} anomalies')"),
        md("## 3. 机器学习异常检测（多维）"),
        code("from anomaly import MLAnomalyDetector\n"
             "md = prepare_all()['multi_dim_sales']\n"
             "X = pd.get_dummies(md[['model','region','channel']])\n"
             "X['sales'] = md['sales'].values\n\n"
             "det = MLAnomalyDetector('isolation_forest', contamination=0.03).fit_predict(X)\n"
             "print(f'Isolation Forest: {int(det[\"is_anomaly\"].sum())} anomalies')\n"
             "print(det[det['is_anomaly']].head())"),
        md("## 4. 人工规则 → 算法方案升级示例\n\n"
           "传统人工规则：`春节月销量 * 0.6`、`车型切换月 * 0.4`。\n\n"
           "算法化升级：用 STL 分解自动捕捉季节性，用 Isolation Forest 检测多维异常。"),
        code("# 对比：人工规则 vs 算法\n"
             "rule_anomalies = md[md['month'].dt.month.isin([1,2])].index  # 人工：1-2月异常\n"
             "print(f'人工规则检出: {len(rule_anomalies)} 条')\n\n"
             "if_det = MLAnomalyDetector('isolation_forest', contamination=0.05).fit_predict(X)\n"
             "print(f'算法检出: {int(if_det[\"is_anomaly\"].sum())} 条')\n"
             "print(f'算法额外发现非春节异常: {(if_det[\"is_anomaly\"] & ~md.index.isin(rule_anomalies)).sum()} 条')"),
        md("## 小结\n\n"
           "- 统计方法适合单指标快速筛查\n"
           "- STL/CUSUM 适合时序季节性与漂移检测\n"
           "- Isolation Forest 可发现人工规则遗漏的多维异常模式"),
    ],
)


# ============================================================
# 04. 智能归因
# ============================================================
save_nb(
    "04_attribution_analysis.ipynb",
    [
        md("# 04. 智能归因分析\n\n"
           "职责3-智能归因：核心指标波动时，通过多维下钻和归因分析定位根因。\n\n"
           "涵盖：单维贡献度、多维下钻、加性分解/Shapley、假设检验显著性验证。"),
        md("## 1. 自动检测异常波动期"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "import pandas as pd\n"
             "from utils.data_loader import prepare_all\n"
             "from attribution import detect_anomaly_period, dimension_contribution, multi_dim_drill_down\n\n"
             "md = prepare_all()['multi_dim_sales'].copy()\n"
             "md['period'] = md['month'].dt.strftime('%Y-%m')\n"
             "periods = detect_anomaly_period(md, 'sales', 'period', threshold=0.2)\n"
             "print('异常波动期:', periods)"),
        md("## 2. 单维度贡献度"),
        code("base, current = periods\n"
             "for dim in ['region', 'model', 'channel']:\n"
             "    c = dimension_contribution(md, dim, 'sales', current, base)\n"
             "    print(f'\\n=== {dim} ===')\n"
             "    print(c.head(3).to_string(index=False))"),
        md("## 3. 多维逐层下钻"),
        code("drill = multi_dim_drill_down(md, ['region','channel','model'], 'sales', current, base, top_n=3)\n"
             "display(drill)"),
        md("## 4. 贡献度分解"),
        code("from attribution import additive_decomposition, waterfall_data\n"
             "add = additive_decomposition(md, ['region','channel'], 'sales', current, base)\n"
             "print('Top 5 贡献组合:')\n"
             "print(add.head(5).to_string(index=False))\n\n"
             "# 瀑布图数据\n"
             "wf = waterfall_data(md, 'region', 'sales', current, base)\n"
             "print('\\n瀑布图数据:')\n"
             "print(wf.to_string(index=False))"),
        md("## 5. 假设检验 — 显著性验证"),
        code("from attribution import HypothesisTester\n"
             "ht = HypothesisTester(alpha=0.05)\n\n"
             "# 检验华南区域 当期 vs 基期 是否显著差异\n"
             "a = md[(md['region']=='华南') & (md['period']==base)]['sales']\n"
             "b = md[(md['region']=='华南') & (md['period']==current)]['sales']\n"
             "print(ht.t_test(a, b))\n"
             "print(ht.mann_whitney(a, b))"),
        md("## 小结\n\n"
           "- 自动检测波动期 + 多维下钻定位根因切片\n"
           "- 贡献度分解量化各维度的贡献占比\n"
           "- 假设检验确保归因结论具有统计显著性，避免过度解读"),
    ],
)


# ============================================================
# 05. 因果推断
# ============================================================
save_nb(
    "05_causal_inference.ipynb",
    [
        md("# 05. 因果推断（JD 加分项）\n\n"
           "超越相关性，识别指标波动的因果根因。\n\n"
           "涵盖：Granger 因果检验、DoWhy 因果效应估计。"),
        md("## 1. Granger 因果检验\n\n"
           "检验广告投放(ad_spend)是否对销量(sales)有预测作用。"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "from utils.data_loader import prepare_all\n"
             "from attribution import GrangerCausality\n\n"
             "daily = prepare_all()['daily_sales'].set_index('date')\n"
             "gc = GrangerCausality(maxlag=7)\n"
             "print(gc.test(daily, 'sales', 'ad_spend'))\n"
             "print(gc.test(daily, 'sales', 'is_weekend'))"),
        md("## 2. 多变量 Granger 因果扫描"),
        code("candidates = ['ad_spend', 'is_weekend', 'is_holiday']\n"
             "result = gc.test_all(daily, 'sales', candidates)\n"
             "display(result)"),
        md("## 3. DoWhy 因果效应估计\n\n"
           "估计广告投放对销量的平均因果效应（ATE）。"),
        code("try:\n"
             "    from attribution import DoWhyCausalEstimator\n"
             "    df = daily.reset_index()\n"
             "    # 二值化处理变量：高投放 vs 低投放\n"
             "    df['high_ad'] = (df['ad_spend'] > df['ad_spend'].median()).astype(int)\n"
             "    est = DoWhyCausalEstimator(\n"
             "        treatment='high_ad',\n"
             "        outcome='sales',\n"
             "        common_causes=['is_weekend', 'is_holiday']\n"
             "    )\n"
             "    result = est.estimate(df)\n"
             "    print(result)\n"
             "except ImportError:\n"
             "    print('dowhy 未安装，请运行: pip install dowhy')\n"
             "    print('DoWhy 可估计因果效应并通过安慰剂检验验证稳健性。')"),
        md("## 小结\n\n"
           "- Granger 因果：时序归因的线索，判断变量间预测关系\n"
           "- DoWhy：基于因果图的效应估计，配合反驳检验确保稳健\n"
           "- 因果推断是归因分析的「升级版」，从相关走向因果"),
    ],
)


# ============================================================
# 06. LLM Agent 报告
# ============================================================
save_nb(
    "06_llm_agent_report.ipynb",
    [
        md("# 06. LLM Agent — 自动化分析报告\n\n"
           "职责4-AI 探索：探索大模型在数据分析场景的应用，参与 AI Agent 产品设计。\n\n"
           "涵盖：可插拔 LLM 接口、自动报告生成、端到端 Agent 编排。"),
        md("## 1. 可插拔 LLM 接口\n\n"
           "设计 `BaseLLMProvider` 抽象接口，支持 Mock/Demo 模式，预留 OpenAI 等接入点。"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "from llm_agent import get_provider, MockLLMProvider, ReportGenerator\n\n"
             "# Mock LLM（无需 API Key）\n"
             "llm = get_provider('mock')\n"
             "print('Provider:', llm.name, '| available:', llm.is_available())\n\n"
             "resp = llm.chat('分析销量异常的原因')\n"
             "print(resp.content[:300])"),
        md("## 2. 自动报告生成"),
        code("from utils.data_loader import prepare_all\n"
             "from attribution import dimension_contribution\n"
             "from anomaly import StatisticalAnomalyDetector\n"
             "import pandas as pd\n\n"
             "ds = prepare_all()\n"
             "byd = ds['byd_monthly']\n"
             "reporter = ReportGenerator(MockLLMProvider())\n\n"
             "# 异常检测报告\n"
             "series = byd.set_index('month')['sales']\n"
             "det = StatisticalAnomalyDetector('sigma', k=2).detect(series)\n"
             "report = reporter.generate_anomaly_report(series, det)\n"
             "print(report)"),
        md("## 3. 端到端 Agent 编排\n\n"
           "数据 → 探索 → 建模 → 异常 → 归因 → 报告 全流程自动化。"),
        code("from llm_agent import AnalysisAgent, MockLLMProvider\n\n"
             "md = ds['multi_dim_sales'].copy()\n"
             "md['period'] = md['month'].dt.strftime('%Y-%m')\n"
             "agent = AnalysisAgent(\n"
             "    df=md, metric_col='sales', period_col='period', time_col='month',\n"
             "    llm=MockLLMProvider()\n"
             ")\n"
             "report = agent.run_all()\n"
             "display(agent.summary())\n"
             "print('\\n=== 综合报告 ===')\n"
             "print(report[:500])"),
        md("## 4. 接入真实 LLM\n\n"
           "设置环境变量后即可切换到真实模型：\n"
           "```python\n"
           "import os\n"
           "os.environ['OPENAI_API_KEY'] = 'your-key'\n"
           "from llm_agent import get_provider\n"
           "llm = get_provider('openai', model='gpt-4o-mini')\n"
             "```"),
        md("## 小结\n\n"
           "- 可插拔设计：Mock 模式离线可用，生产环境接入真实 LLM\n"
           "- Agent 编排：端到端自动化，减少人工分析环节\n"
           "- 报告生成：将结构化分析结果转化为自然语言业务报告"),
    ],
)


# ============================================================
# 07. pro-dataset 端到端
# ============================================================
save_nb(
    "07_pro_dataset_end_to_end.ipynb",
    [
        md("# 07. 火山引擎专业数据集 — 端到端示例\n\n"
           "本 Notebook 演示如何通过火山引擎「专业数据集」MCP 工具获取真实业务数据，并完成端到端分析。\n\n"
           "> 注：运行本 Notebook 需要在 TRAE 中授权专业数据集插件。离线环境下可直接使用 `data/processed/` 中已缓存的真实数据。"),
        md("## 1. 调用专业数据集 MCP 工具\n\n"
           "在 TRAE 环境中，通过 `dataPro_search` 工具检索数据。以下为已获取的真实数据结构示例："),
        code("import json\n"
             "# 示例：已获取的比亚迪月度销量（来自 vehicle_sales 数据集）\n"
             "sample = {\n"
             "  'dataset_type': 'vehicle_sales',\n"
             "  'items': [{\n"
             "    '查询条件': {'品牌': '比亚迪', '时间范围': '近一年', '地域': '全国'},\n"
             "    '销售数据': [\n"
             "      {'时间': '2025-09', '销量': 313167, '在售厂商份额': '13.86%'},\n"
             "      {'时间': '2026-08', '销量': 183789, '在售厂商份额': '11.88%'},\n"
             "    ]\n"
             "  }]\n"
             "}\n"
             "print(json.dumps(sample, ensure_ascii=False, indent=2))"),
        md("## 2. 加载已缓存的真实数据"),
        code("import sys; sys.path.insert(0, '../src')\n"
             "from utils.data_loader import load_byd_monthly, load_tesla_monthly, load_byd_model_aug\n"
             "import pandas as pd\n\n"
             "byd = load_byd_monthly()\n"
             "tesla = load_tesla_monthly()\n"
             "model_aug = load_byd_model_aug()\n\n"
             "print('比亚迪月度:'); display(byd.head())\n"
             "print('比亚迪车型(8月):'); display(model_aug)"),
        md("## 3. 端到端分析：比亚迪 vs 特斯拉"),
        code("import matplotlib.pyplot as plt\n"
             "fig, ax = plt.subplots(figsize=(12, 5))\n"
             "byd.plot(x='month', y='sales', ax=ax, label='比亚迪', marker='o')\n"
             "tesla.plot(x='month', y='sales', ax=ax, label='特斯拉', marker='s')\n"
             "ax.set_title('比亚迪 vs 特斯拉 月度销量对比（真实数据）')\n"
             "ax.legend(); plt.tight_layout()"),
        md("## 4. 车型结构分析"),
        code("model_aug.groupby('network')['sales'].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8,8))\n"
             "plt.title('比亚迪 2026年8月 各网络销量占比')"),
        md("## 5. 接入你的数据\n\n"
           "如需获取最新数据，在 TRAE 中授权后运行：\n"
           "```python\n"
           "# 在 TRAE Agent 中调用\n"
           "result = dataPro_search({'query': '比亚迪 近一年全国销量趋势'})\n"
           "# 将结果保存后用 utils.data_loader 中的格式加载\n"
             "```"),
        md("## 小结\n\n"
           "- 专业数据集提供真实可信的业务数据\n"
           "- 本项目已缓存比亚迪/特斯拉真实销量作为示例\n"
           "- 授权后可获取最新数据，替换缓存数据即可"),
    ],
)

print(f"\n共生成 {len(list(NB_DIR.glob('*.ipynb')))} 个 Notebook")
