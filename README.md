<p align="center">
  <img src="frontend/logo/logo_big.png" alt="AutoSTAT Logo" width="180">
</p>

<h1 align="center">AutoSTAT v2</h1>

<p align="center">
Statistical analysis, instantly. AutoSTAT v2 是一个面向数据分析全流程的多 Agent 框架，覆盖数据导入、预处理、可视化、建模和报告生成。
</p>

---

## 功能特点

- **端到端分析流程**：从 CSV 数据导入到 Word / HTML / Markdown 报告导出，串联完整统计分析链路。
- **模块化 Agent 设计**：Planning、Loading、Preprocessing、Visualizing、Modeling、Reporting 分工协作，便于调试和扩展。
- **代码生成与自愈**：预处理、可视化和建模阶段支持 LLM 生成代码，并在执行失败时自动修复。
- **报告图文匹配**：报告生成阶段会基于目录、图表标题和图表上下文规划插图位置，减少漏图、错图和重复图。
- **灵活 LLM 后端**：支持 OpenAI 兼容 API，如 DeepSeek、OpenAI、通义千问、Moonshot 和本地 Ollama。
- **轻量级 RAG**：内置 BM25 算法知识库检索，无需额外部署向量数据库。

---

## 快速开始

请先安装 Python 3.10 及以上版本。

### 1. 克隆项目

```bash
git clone https://github.com/Jiaye-s-Group/AutoSTAT-ver2.git
cd AutoSTAT-ver2
```

### 2. 创建环境

```bash
conda create --name autostat python=3.12
conda activate autostat
pip install -r requirements.txt
```

### 3. 配置 LLM

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-你的key
OPENAI_MODEL=deepseek-chat
```

常用配置示例：

| 服务 | `OPENAI_BASE_URL` | `OPENAI_MODEL` |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| Ollama | `http://localhost:11434/v1` | `qwen2.5` |

### 4. 启动应用

```bash
streamlit run app.py
```

默认访问地址为 `http://localhost:8501`。

---

## 使用方式

### Web UI

按页面顺序完成分析：

1. **数据导入**：上传 CSV，解析字段含义。
2. **数据预处理**：生成预处理建议和代码，得到清洗后的数据。
3. **数据可视化**：生成可视化推荐、图表代码和图表解读。
4. **建模分析**：设置目标列，生成并执行建模代码，输出模型结果。
5. **报告生成**：生成目录，逐节撰写报告，并导出 Word / HTML / Markdown。

也可以在侧边栏开启自动模式，一键串联完整流程。

### CLI

```bash
python -m workflows.planning data.csv "分析目标"
python -m workflows.loading data.csv
python -m workflows.preprocessing data.csv
python -m workflows.visualizing data.csv
python -m workflows.modeling data.csv target_column
python -m workflows.autostat data.csv target_column
```

### Python 调用

```python
import pandas as pd
from workflows.autostat import run_autostat

df = pd.read_csv("data.csv")
result = run_autostat(
    df,
    user_input_model="我想做销量预测",
    target_column="sales",
)
print(result["final_html"][:500])
```

---

## 项目结构

```text
AutoSTAT-ver2/
├── app.py                  # Streamlit 启动入口
├── requirements.txt        # Python 依赖
├── core/                   # LLM、Prompt、RAG 和 workflow 基础工具
├── workflows/              # 本地 workflow 实现
├── prompts/                # 各阶段 LLM Prompt 模板
├── knowledge/              # 算法知识库
├── frontend/               # Streamlit 前端页面与导出工具
└── _extracted/             # workflow 结构说明与图谱文件
```

---

## 常见问题

**Q: Loading workflow 报 `Insufficient Balance`？**

这是当前 LLM API Key 所属账户余额不足导致的 402 错误。请更换有余额的 API Key，或切换到其他 OpenAI 兼容服务。

**Q: 提示 `response_format is not supported`？**

系统会自动降级为普通文本解析 JSON，通常不影响使用。

**Q: 生成代码一直失败？**

可以尝试更强的模型，或调整对应阶段的 `prompts/` 模板。系统会在代码执行失败时自动进行多轮修复。

**Q: 如何替换算法知识库？**

可以编辑 `knowledge/algorithm_catalog.jsonl`，每行一条 JSON 记录。

**Q: Windows 下出现编码问题？**

```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
```

---

## 许可

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE)。
