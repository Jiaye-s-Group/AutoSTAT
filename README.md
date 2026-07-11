<p align="center">
  <img src="frontend/logo/github.jpeg" alt="Automated Statistician" width="100%">
</p>

<h1 align="center">AutoSTAT v2</h1>

<p align="center">
  <strong><em>Statistical Analysis, Instantly.</em></strong>
</p>

<p align="center">
  <a href="./docs/ARCHITECTURE.md">架构说明</a> ·
  <a href="./docs/CONTRIBUTING.md">贡献指南</a>
</p>

AutoSTAT v2，致力于成为用户数据分析的 copilot。

我们希望构建一个入门友好、覆盖数据分析端到端流程、能够通过自然语言交互持续优化分析效果，并适配未来 LLM 能力迭代的数据分析 Agent 框架。用户只需上传数据、描述分析目标，AutoSTAT 就可以协助完成数据导入、预处理、可视化、建模和报告生成等核心任务。

---

## v2 主要更新

- **引入 RAG 方法库检索。** v2 内置统计与机器学习方法知识库，在预处理、可视化和建模代码生成前检索相关方法参考，为生成代码提供更稳定的算法依据。
- **升级为完整本地工作流。** v2 将 Planning、Loading、Preprocessing、Visualizing、Modeling 和 Reporting 拆分为清晰的 workflow 模块，各阶段结果会继续进入后续分析链路。
- **增强代码生成与自动修复。** 预处理、可视化和建模阶段支持生成可执行 Python 代码，并在运行失败时结合报错信息进行多轮修复。
- **强化报告生成能力。** 报告阶段会综合数据概览、预处理结果、图表分析、模型结果和用户需求，生成结构化分析报告，并支持预览、进度显示和多格式导出。
- **支持多模型与自定义接口。** 支持 DeepSeek、OpenAI、Qwen、ZhipuAI、Doubao、AIHubMix，以及任意 OpenAI-compatible API。
- **支持上传参考资料。** 用户可以上传 PDF、DOCX、TXT 等参考资料，系统会在分析过程中检索相关内容辅助生成结果。

---

## 功能特点

- **全流程覆盖，重构数据分析体验。** AutoSTAT v2 覆盖导入、预处理、可视化、建模与报告生成五个核心流程，让用户可以从原始数据一路走到可交付报告。
- **模块化 Agent 与工作流设计。** 每个分析阶段都有独立职责，既便于用户理解流程，也便于开发者调试、替换和扩展。
- **代码生成，释放数据分析潜能。** 系统不仅能调用固定分析逻辑，也能根据用户需求生成新的 Python 分析代码，兼顾稳定性和灵活性。
- **自动模式，一键推进完整分析。** 面向非技术用户，上传数据后可开启自动模式，由 Planning 模块判断任务路径并串联后续阶段。
- **专业报告，沉淀最终分析成果。** 报告生成阶段会整合图表、模型结果和分析上下文，输出更接近可直接使用的图文报告。

---

## 快速开始

请先安装 Python 3.10 及以上版本，推荐 Python 3.11 或 3.12。支持 macOS、Windows 和 Linux。

### 1. 克隆项目

```bash
git clone https://github.com/Jiaye-s-Group/AutoSTAT.git
cd AutoSTAT
```

### 2. 创建环境

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 配置模型

可以直接在应用侧边栏配置模型，也可以复制 `.env.example`：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
copy .env.example .env
```

编辑 `.env`：

```ini
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-你的 API Key
OPENAI_MODEL=deepseek-chat
```

常用模型配置示例：

| 服务 | `OPENAI_BASE_URL` |
|---|---|
| DeepSeek | `https://api.deepseek.com/v1` |
| OpenAI | `https://api.openai.com/v1` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` |
| 豆包 | `https://ark.cn-beijing.volces.com/api/v3` |
| AIHubMix | `https://aihubmix.com/v1` |
| 自定义接口 | 你的 OpenAI-compatible 地址 |

如果选择在侧边栏保存配置，配置会写入 `~/.config/autostat/config.toml`，不会写入项目目录。

### 4. 启动应用

```bash
streamlit run app.py
```

默认访问地址为 `http://localhost:8501`。

---

## 使用方式

### Web UI

按页面顺序完成分析：

1. **数据导入**：上传数据文件，查看字段、类型和数据预览。
2. **数据预处理**：生成预处理建议和代码，得到清洗后的数据。
3. **数据可视化**：生成图表建议、可视化代码和图表解读。
4. **建模分析**：选择目标列，生成建模方案和训练结果。
5. **报告生成**：生成目录、撰写正文，并导出 HTML、Markdown、Word 或 PDF。

也可以在侧边栏开启自动模式，一键串联完整分析流程。

### 样例数据

首次体验可以上传：

```text
examples/tiny_sales.csv
```

该样例包含日期、区域、渠道、收入、订单数和退货率，适合快速测试数据预览、可视化、建模和报告生成流程。

---

## 项目结构

```text
AutoSTAT/
├── app.py                  # Streamlit 启动入口
├── requirements.txt        # Python 依赖
├── core/                   # LLM、配置、Prompt、RAG 等基础能力
├── workflows/              # 各阶段本地 workflow 实现
├── frontend/               # Streamlit 前端页面、状态与导出工具
├── prompts/                # 各阶段 LLM Prompt 模板
├── knowledge/              # 内置方法知识库
├── examples/               # 示例数据
└── docs/                   # 架构说明与贡献指南
```

---

## 开发说明

AutoSTAT v2 遵循清晰的代码边界：

- `frontend/` 负责用户交互、页面展示和 Streamlit 状态。
- `workflows/` 负责分析逻辑，每个阶段接收普通 Python 输入并返回结构化结果。
- `core/` 负责可复用基础能力，包括 LLM 调用、模型配置、Prompt 渲染和检索。

更多说明见：

- [架构说明](./docs/ARCHITECTURE.md)
- [贡献指南](./docs/CONTRIBUTING.md)

本地开发检查：

```bash
pip install -e ".[dev]"
ruff check .
```

---

## 常见问题

**Q: 没有 OpenAI API Key，可以使用其他模型吗？**

可以。只要服务提供 OpenAI-compatible Chat Completions API，就可以通过 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `OPENAI_MODEL` 接入。

**Q: API Key 会保存在哪里？**

默认读取 `.env` 或系统环境变量。如果在侧边栏选择保存，会写入用户目录 `~/.config/autostat/config.toml`，不会保存到项目仓库。

**Q: 为什么代码生成会自动重试？**

AutoSTAT 会在本地执行生成的分析代码。如果代码运行失败，系统会把错误信息反馈给模型，并尝试自动修复。

**Q: 如何替换内置方法知识库？**

可以编辑 `knowledge/algorithm_catalog.jsonl`，每行是一条 JSON 记录。

---

## 许可

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。
