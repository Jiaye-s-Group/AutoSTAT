# 第 ② 批交付说明

## 本批产出

| 模块 | 说明 | 状态 |
|---|---|---|
| `core/llm_client.py` | OpenAI 兼容 LLM 客户端（默认 deepseek-chat） | ✅ |
| `core/prompt_template.py` | `{{var}}` 渲染 + 缺失变量检查 | ✅ |
| `core/rag_retriever.py` | 基于黄页的 BM25 检索 | ✅ |
| `core/workflow_runner.py` | 5 个辅助函数（safe_object / dig / to_str 等） | ✅ |
| `workflows/_plugins.py` | 16 个 Coze plugin 的本地实现 | ✅ |
| `workflows/planning.py` | Planning workflow + CLI | ✅ |
| `workflows/loading.py` | Loading workflow + CLI | ✅ |
| `.env.example` | 环境变量模板 | ✅ |
| `requirements.txt` | 依赖清单 | ✅ |

## 离线验证（我已经跑通的）

```
✓ 所有 core 模块导入成功（openai 懒加载）
✓ 全部 72 个 prompt 文件可渲染
✓ RAG 检索精确命中（"缺失值处理算法 均值填充 K邻近填充" → 命中 3 条，top-1 得分 1.0）
✓ 16 个 plugin 全部可调用
✓ code_runner 真实在子进程里跑通了 "process_df = df.fillna(df.mean())"
✓ code_runner 错误捕获正常（错误代码不会让主进程崩）
✓ parse_json_best_effort 兼容 ```json 包装 / 多余文字的 LLM 输出
```

## 使用方式

### 1. 安装依赖

```bash
cd autostat_local
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
# 默认走 DeepSeek；要换模型改 OPENAI_BASE_URL + OPENAI_MODEL 即可
```

### 3. 测试 Planning workflow（独立运行）

```bash
python -m workflows.planning path/to/your/data.csv "我想做房价预测"
```

预期输出：
```
loading_auto:  True
prep_auto:     True
vis_auto:      True
modeling_auto: True
report_auto:   True

----- plan -----
（LLM 生成的分析路径说明...）
```

### 4. 测试 Loading workflow（独立运行）

```bash
python -m workflows.loading path/to/your/data.csv "我想做房价预测"
```

预期输出：
```
===== summary_1.title =====
数据概览与数据含义分析

===== summary_1.desc =====
（LLM 生成的数据含义分析...）

===== abstract_1 =====
（一段式摘要...）
```

## 设计要点

### LLM 客户端（llm_client.py）
- **`chat(sys, user)`** — 普通文本输出
- **`chat_json(sys, user)`** — 强制 JSON（自动兼容不支持 `response_format` 的模型，降级到"从文本里挖 JSON"）
- **懒加载 openai** — 没装 openai 也能 import core.rag_retriever 做检索测试
- **重试** — 失败时自动重试，便于调试和稳定运行

### 模板渲染（prompt_template.py）
- Coze 的 `{{var}}` 语法直接兼容
- 找不到的变量渲染为空串（防 `"None"` 出现在 prompt 里）
- 支持 `{{obj.key}}` 深层取值
- 自动处理 dict/list → JSON 字符串

### RAG 检索（rag_retriever.py）
- **BM25 + 字段加权**：`name` / `category_l2` 字段权重 ×3（因为 get_query LLM 就按它们拼 query）
- **可选 jieba 分词**：装了用 jieba，没装用正则兜底
- **归一化打分**：top-1 始终是 1.0，方便用 `min_score` 过滤
- 274 条算法 × 5 字段，**冷启动 < 100ms**

### Plugin 模块（workflows/_plugins.py）
- 16 个 plugin 按组织：Composer / 列表处理 / 数据加载 / 统计 / 代码执行 / 可视化 / RAG 格式化
- **`code_runner`** 用子进程隔离（超时+错误捕获），LLM 生成的代码就算炸也不会带崩主程序
- **`df_to_meta`** 是 `Loading_Data` plugin 的本地版——直接从 DataFrame 生成元信息，不再需要 URL

## 下一批预告

第 ③ 批：
- `workflows/preprocessing.py`（含 Loop + RAG）
- `workflows/visualizing.py`（含 2 个 Batch 节点）

第 ④ 批：
- `workflows/modeling.py`（含 RAG + 最复杂的 9 个 LLM 节点）

第 ⑤ 批：
- `workflows/reporting_toc.py` + `workflows/reporting_partly.py`（含 final_html 数组 join）
- `workflows/autostat.py`（总编排）
- **前端 render 页面替换**（把原 call_coze_workflow 改成本地 run_xxx_workflow，session_state 零改动）

## 调试提示

如果你跑 `python -m workflows.loading xxx.csv` 发现 LLM 输出质量不好：

1. 查看前端报错信息与 traceback 输出，定位当前调用上下文
2. **改用更强的模型**：把 `.env` 里的 `OPENAI_MODEL` 改成 `deepseek-reasoner` 或 `gpt-4o`
3. **调 prompt**：直接改 `prompts/loading/xxx.txt` 里的文件，不用改代码
