# AutoSTAT 本地化 Agent — 最终版

把原本跑在 Coze 云上的 8 个 workflow 完全"vibe"成本地 Python 代码。
Streamlit 前端保留原样，调 LLM 改走你自选的 OpenAI 兼容接口。

---

## ⚡ 最快上手（3 步）

```bash
cd autostat_local

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 3. 启动前端
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，能看到和原版一模一样的 Streamlit UI。

---

## 🎯 目录结构

```
autostat_local/
├── .env.example           # 环境变量模板（填 DeepSeek / OpenAI / 通义 / Kimi 等）
├── requirements.txt
├── README.md              ← 你正在看
│
├── core/                  # 核心基础设施
│   ├── llm_client.py      #   OpenAI 兼容客户端（懒加载、JSON 鲁棒 parse、重试）
│   ├── prompt_template.py #   {{var}} 渲染
│   ├── rag_retriever.py   #   基于算法黄页的 BM25 检索
│   └── workflow_runner.py #   safe_object / dig / as_bool 等辅助函数
│
├── workflows/             # 8 个 workflow 的本地 Python 实现
│   ├── _plugins.py        #   16 个自定义 Coze plugin 的本地版
│   ├── planning.py        #   规划 5 个阶段的开关
│   ├── loading.py         #   数据语义解析
│   ├── preprocessing.py   #   预处理（含 Loop 修复 + RAG）
│   ├── visualizing.py     #   可视化（含 2 个并发 Batch）
│   ├── modeling.py        #   建模（含 RAG + Loop 修复）
│   ├── reporting_toc.py   #   生成报告目录
│   ├── reporting_partly.py #  按目录逐节写，final_html 数组 join
│   └── autostat.py        #   总编排（串联 6 个子 workflow）
│
├── prompts/               # 72 个 LLM prompt（原样从 Coze JSON 抽出）
│   ├── planning/  loading/  preprocessing/  visualizing/
│   └── modeling/  reporting_toc/  reporting_partly/
│
├── knowledge/             # RAG 知识库
│   ├── 算法黄页.xlsx       #   你提供的原文件（274 条算法）
│   ├── algorithm_catalog.jsonl  # 规范化后的检索版
│   └── category_tree.json
│
├── code_nodes/            # Coze Code 节点的原始 Python 代码（参考用）
├── _extracted/            # workflow 结构化数据（FLOW.md + graph.json）
│
├── frontend/              # Streamlit 前端（保留原版 UI，只替换调用层）
│   ├── app.py             #   主入口
│   ├── utils/
│   │   ├── coze_runtime.py         # stub（兼容旧代码调用）
│   │   └── local_workflow_bridge.py # ⭐ 前端→本地 workflow 的桥接
│   └── workflow/          #   6 个页面 render（call_coze_* 已替换为本地调用）
│       ├── dataloading/   visualization/
│       ├── preprocessing/ modeling/
│       ├── report/        preference/
│
└── tests_offline*.py      # 离线自检脚本（不调 LLM）
```

---

## 🚀 完整运行说明

### 第 1 步 · 装依赖

```bash
cd autostat_local
pip install -r requirements.txt
```

**包括** `streamlit / openai / pandas / plotly / scikit-learn / python-docx` 等。
**不依赖** `cozepy`、`requests` 调 Coze。

### 第 2 步 · 配置 LLM

`.env.example` 默认 DeepSeek（便宜稳定）：

```ini
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-你的key
OPENAI_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

支持替换为任何 OpenAI 兼容接口：

| 服务 | BASE_URL | MODEL |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5` |

```bash
cp .env.example .env
# 编辑 .env 填 API key
```

### 第 3 步 · 启动方式（三选一）

#### ① 完整前端（推荐）

保留原 Streamlit UI，5 个页面逐一走：

```bash
streamlit run app.py
```

浏览器打开后：
1. **数据导入页** — 上传 CSV → 点"解析含义" → 本地 Loading workflow 跑
2. **数据预处理页** — 获取建议 → 生成代码 → 自愈执行
3. **数据可视化页** — 推荐方案 → 生成 plotly 代码 → 批量生成图+分析
4. **建模分析页** — 推荐模型 → 训练代码 → 自愈执行
5. **报告生成页** — 生成目录 → 逐节填报告 → 下载 Word/HTML/MD/PDF

也可以点侧边栏的 **"🚗 开启自动模式"** 一键跑完所有页面。

#### ② CLI 单跑某个 workflow

```bash
# 测 Planning
python -m workflows.planning data/iris.csv "我想做鸢尾花分类预测"

# 测 Loading
python -m workflows.loading data/iris.csv

# 测 Preprocessing（含代码生成+执行）
python -m workflows.preprocessing data/iris.csv

# 测 Visualizing（含 Batch 并发）
python -m workflows.visualizing data/iris.csv

# 测 Modeling（需指定 target 列）
python -m workflows.modeling data/iris.csv species

# 一键跑完整条链路
python -m workflows.autostat data/iris.csv species
# 生成 autostat_report.html
```

#### ③ 离线自检（不花 token）

```bash
python tests_offline.py         # 第②批：core + plugins
python tests_offline_batch3.py  # 第③批：_batch_run + prompt 渲染
```

---

## 🔑 关键技术决策

### 1. LLM 调用层：OpenAI 兼容 + 懒加载

`core/llm_client.py` 封装 `chat()` / `chat_json()`，支持：
- 自动重试 2 次
- 不支持 `response_format` 的模型自动降级，从文本里 parse JSON

### 2. 代码自愈循环（最多 5 次）

Preprocessing / Visualizing / Modeling 都内置：
```
code_runner(执行) → 失败 → Code_Fixer LLM(修) → 再执行 → ... → 最多 5 次
```

5 次仍失败则**退化**：Preprocessing 原 df 透传、Visualizing 返回空图表列表并给出错误信息。**不会因为 LLM 生成了坏代码整个流程炸掉**。

### 3. 子进程 + 超时隔离

所有用户代码（LLM 生成的 plotly/sklearn 代码）都在**独立 Python 子进程**里跑：
- 死循环 → 超时杀掉（默认 300 秒）
- 抛异常 → 父进程捕获，不影响 Streamlit

### 4. RAG 检索（无向量库）

`core/rag_retriever.py` 基于算法黄页做 **BM25 + 字段加权**（`name`×3 + `category_l2`×3）：
- 274 条算法，冷启动 < 100ms
- 查询 `"缺失值处理算法 均值填充 K邻近填充"` → top-1 score=1.0（精确命中）
- 可选 `jieba` 分词（装了就用，没装走正则）

### 5. Batch 节点 → ThreadPoolExecutor

Coze 原 `batchSize=100, concurrent=10`。本地用线程池 1:1 复刻：
- 实测 10 张图的 LLM 生成并发耗时 ≈ 1 次调用（串行会 > 10 倍）
- 如果你的 LLM provider 有 QPS 限制，调小 `workflows/visualizing.py` 里的 `BATCH_CONCURRENCY`

### 6. 前端零侵入

保留了原版 Streamlit 前端的 **session_state 结构、页面布局、UI 交互**。替换只在两处：
- `utils/coze_runtime.py` → stub，兼容旧调用
- `utils/local_workflow_bridge.py` → 把本地 workflow 包装成 Coze 返回结构

render 页面里只替换了 `call_coze_workflow*` 函数体，**其他 3000+ 行 UI 代码全部没动**。

### 7. Reporting 的 final_html 拼接

你强调的"逐部分输出需要拼接后输出"：
- `workflows/reporting_partly.py` 里 `report_parts` 是 list
- **拼接点**：`final_html = "".join(report_parts)` （批式）
- 同时返回 `final_html_parts` 数组，方便未来做流式展示

---

## 🛠 常见问题

### Q: 跑起来 LLM 特别慢？
A: 可能是 RAG 关掉后用了更复杂的模型。把 `OPENAI_MODEL` 改成 `deepseek-chat` 或 `gpt-4o-mini`。

### Q: LLM 报 `response_format is not supported`？
A: `chat_json()` 会自动降级到"从文本挖 JSON"模式。本地 Ollama / 旧版模型都不影响。

### Q: 生成的代码跑不通？
A: 系统自动会最多重试 5 次。如果一直失败：
1. 考虑换更强的 `OPENAI_MODEL`
2. 直接改 `prompts/preprocessing/code_generation_llm_sys.txt` 优化 prompt

### Q: 不想用前端，只想当库调？
A: 直接 `from workflows.autostat import run_autostat`：
```python
import pandas as pd
from workflows.autostat import run_autostat

df = pd.read_csv("data.csv")
result = run_autostat(
    df,
    user_input_model="我想做销量预测",
    target_column="sales",
    on_step=lambda step, _: print(f"完成 {step}"),
)
print(result["final_html"][:500])
```

### Q: 我的算法黄页想换？
A: 直接替换 `knowledge/算法黄页.xlsx`，或修改 `knowledge/algorithm_catalog.jsonl`。
格式：每行一个 JSON `{id, category_l1, category_l2, name, description, code}`。

### Q: 遇到乱码/编码错误？
A: 所有文件都用 UTF-8。Windows PowerShell 跑 CLI 时建议：
```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
```

---

## 📊 核心数据流（示意）

```
┌──────────┐
│ CSV 文件  │
└────┬─────┘
     ▼
┌──────────────────────────────────┐
│ Planning (2 LLM)                  │  plan: 5个开关
│ df → meta → planner LLM(JSON) →   │  + shape, dtype_info_str
│  analysis_path LLM                │    head_dict_str, df
└────┬─────────────────────────────┘
     ▼
┌──────────────────┐
│ Loading (3 LLM)   │  summary_1, abstract_1
└────┬─────────────┘
     ▼
┌────────────────────────────┐
│ Preprocessing (6 LLM + RAG) │  summary_2, abstract_2, suggestion
│ + Loop(5 次修复)            │  processed_df → 下游用
└────┬───────────────────────┘
     ▼
┌──────────────────────────────┐
│ Visualizing (8 LLM + 2 Batch) │  summary_3, abstract_3,
│ + Loop(5 次修复)              │  final_code, tu_title
└────┬────────────────────────┘
     ▼
┌───────────────────────┐
│ Modeling (9 LLM + RAG) │  summary_4, abstract_4, model_suggestion
│ + Loop(5 次修复)       │
└────┬──────────────────┘
     ▼
┌────────────────────────────────┐
│ Reporting_toc (2 LLM)           │  toc_text + 4 个 abstract 透传
└────┬───────────────────────────┘
     ▼
┌─────────────────────────────────────────┐
│ Reporting_partly (2 LLM + Loop 对每节): │
│   writer → fill → composer → validator   │
│ → final_html_parts: [...]                │
│ → final_html = "".join(parts)  ⭐ 拼接点  │
│ → title_maker LLM                        │
└────┬─────────────────────────────────────┘
     ▼
┌──────────────────────────┐
│ Streamlit 显示 + 下载导出 │
│ (Word / HTML / Markdown) │
└──────────────────────────┘
```

---

## 📋 Check List（确认清单）

### 前端是否成功切换到本地
- [ ] 启动后侧边栏显示 **"状态：本地模式已就绪，LLM=xxx"**（绿色）
- [ ] 侧边栏不再有"Coze 授权"按钮
- [ ] 各页面跑 workflow 时 terminal 里看不到 `api.coze.com` 的请求

### Workflow 是否都能跑通
- [ ] Loading：侧边栏上传 CSV → "解析含义" → 出 summary_1 卡片
- [ ] Preprocessing：出建议 → 生成代码 → 看到"代码执行：✅ 成功"
- [ ] Visualizing：推荐方案 → 生成代码 → 页面显示 N 张图
- [ ] Modeling：选 target → 生成代码 → 出 result_format
- [ ] Reporting：生成目录 → 生成完整报告 → 可下载 Word

---

## 📖 分批交付文档

这个最终版是 5 个批次累积起来的：

- **第①批**：抽取 72 个 prompt + 12 个 code 节点 + 结构化 graph.json
- **第②批**：core 模块 + Planning + Loading workflow
- **第③批**：Preprocessing + Visualizing workflow
- **第④⑤批（本次）**：Modeling + Reporting + AutoSTAT 总编排 + **前端集成**

分批交付的说明文档（`BATCH2_README.md` / `BATCH3_README.md`）仍然保留在包里作为历史记录。

---

有问题欢迎提。想加功能（如流式展示 final_html / 切换多个 RAG 知识库 / 自定义 plugin）都可以在这个架构上直接扩展。
