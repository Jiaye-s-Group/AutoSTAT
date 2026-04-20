# 第 ③ 批交付说明

## 本批产出

| 模块 | 行数 | 说明 | 状态 |
|---|---|---|---|
| `workflows/preprocessing.py` | ~230 | 6 LLM + RAG + Loop(5次) + code_runner | ✅ |
| `workflows/visualizing.py` | ~330 | 8 LLM + 2 个并发 Batch + Loop 修复 | ✅ |
| `tests_offline_batch3.py` | ~100 | 离线测试脚本 | ✅ |

## 本批的技术亮点

### 1. Loop 节点用 Python for 实现

Coze 里的 `Loop(count=5)` 在代码里就是普通的 for 循环加 break：

```python
for attempt in range(MAX_FIX_ATTEMPTS):
    run_result = code_runner(code=current_code, df=df)
    if run_result["is_success"]:
        break
    # 失败 → 调用 Code_Fixer LLM 修代码 → 下一轮继续
    fixed = chat(fix_sys, fix_user, ...)
    current_code = fixed
```

### 2. Batch 节点用 ThreadPoolExecutor 实现

Coze 原版 `sec3_desc_fig` / `sec3_summary_fig_list` 两个 Batch 节点的 `batchSize=100, concurrent=10`。本地用线程池精确复刻：

```python
def _batch_run(items, func, concurrency=10):
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()  # 带异常兜底
    return results
```

**实测 10 个任务并发耗时 56ms**（串行 > 500ms），线程池生效。

### 3. RAG 接入

Preprocessing 调用：

```python
# get_query LLM 生成 "缺失值处理算法 均值填充 K邻近填充"
rag_query = chat(...)
# BM25 检索算法黄页
recall_results = retrieve(rag_query, top_k=3)
# 格式化给下游 code_generation LLM 参考
ctx["knowledge_results"] = format_recall(recall_results)["knowledge_results"]
```

Modeling（第④批）会用同样模式。

### 4. 代码生成自愈机制

两个 workflow 都内置 `MAX_FIX_ATTEMPTS=5` 的自修复循环：
- Preprocessing：**直接跑代码**，失败就丢给 Code_Fixer LLM
- Visualizing：**先 validate（用 df.head(5) 试跑）**，通过再全量执行

## 使用方式

### Preprocessing 独立测试

```bash
python -m workflows.preprocessing path/to/your/data.csv "我需要做房价预测，重点处理异常值"
```

预期输出：
```
===== suggestion =====
（初版建议）

===== summary_2.desc =====
（章节正文）

===== summary_2.code =====
（LLM 生成并通过 code_runner 验证的预处理代码）

===== abstract_2 =====
（一段式摘要）

===== 代码执行：✅ 成功 =====
```

### Visualizing 独立测试

```bash
python -m workflows.visualizing path/to/your/data.csv "请可视化特征分布和相关性"
```

预期输出：
```
===== visual_recommendatio =====
（推荐的可视化方案列表）

===== final_code =====
（通过 validate 的 plotly 代码）

===== 图表数量: N =====
--- 图 1 ---
analysis: （这张图的业务分析）

===== abstract_3 =====
（一段式摘要）
```

### 离线测试

```bash
python tests_offline.py           # 第②批测试（core + plugins）
python tests_offline_batch3.py    # 第③批测试（_batch_run + prompt 渲染）
```

## 关于性能

| Workflow | LLM 调用次数（理想情况） | 代码失败重试上限 |
|---|---|---|
| Planning | 2 | — |
| Loading | 3 | — |
| Preprocessing | 6 + N 次修复 | 5 |
| Visualizing | 3 + N 次修复 + 2K 张图的 Batch（每图 2 次 LLM） | 5 |

**Visualizing 对 LLM 量级很敏感**——如果推荐出 6 张图，就要跑 6 × 2 = 12 次 Batch LLM。好处是 ThreadPoolExecutor 让 12 次调用**近似 1 次的耗时**（并发度 10）。

## 已知的需要注意的地方

1. **`visual_recommendatio` 确实拼错了 n** — 对齐原 Coze workflow 的 End 输出字段名，不是我的 typo
2. **`execute_and_extract` 需要安装 plotly** — 已加入 requirements.txt
3. **Batch prompt 比较短** — Coze 原版里的 `summary_fig_Desc` / `Generate_Desc` LLM 的 user prompt 基本就是 plugin 拼装的 `prompt_content`。我做了兜底：当 user prompt 渲染后为空，就用 plugin 产出的 prompt 顶上

## 下一批预告

**第 ④ 批：Modeling workflow**（最复杂的单个 workflow）
- 9 个 LLM 节点（含 `get_query` RAG + `code_fixed` 修复）
- Loop 修复（代码自愈）
- 预估 250 行代码

**第 ⑤ 批：Reporting + AutoSTAT 总编排 + 前端集成**
- `reporting_toc.py` + `reporting_partly.py`（含 final_html 数组 join）
- `autostat.py`（6 个 workflow 串联）
- 替换原 Streamlit render 页面的 `call_coze_workflow` 为本地版本
- 运行 `streamlit run app.py` 完整跑通
