"""不依赖 openai，做语义/结构检查"""
import sys, os
sys.path.insert(0, '.')

os.environ['OPENAI_API_KEY'] = 'test-key'

print("=" * 60)
print("1. 检查 core 模块导入")
print("=" * 60)
from core.prompt_template import render, render_file, find_missing_vars
from core.workflow_runner import safe_object, dig, to_str, to_json_str, as_bool
from core.rag_retriever import get_retriever, retrieve, format_recall
# llm_client 也应该能 import 了（懒加载 openai）
from core.llm_client import LLMClient, chat, chat_json, parse_json_best_effort
print("✓ 所有 core 模块导入成功（openai 懒加载）")

print("\n" + "=" * 60)
print("2. 检查模板渲染")
print("=" * 60)
ctx = {
    "shape_0": 150, "shape_1": 5,
    "dtype_info_str": '{"sepal_length":"float64"}',
    "head_dict_str": '[{"a":1}]',
    "user_input": "", "add_preference": "", "preference_selected": "",
    "loading_auto": True, "prep_auto": True, "vis_auto": True,
    "modeling_auto": False, "report_auto": True, "preference_select": "",
}
from pathlib import Path
total_prompts = 0
for txt_file in Path('prompts').rglob('*.txt'):
    template = txt_file.read_text(encoding='utf-8')
    missing = find_missing_vars(template, ctx)
    rendered = render(template, ctx)
    total_prompts += 1
print(f"✓ 全部 {total_prompts} 个 prompt 可渲染")

# 深入看 Planning 两个 prompt 的 missing 情况
for f in ['prompts/planning/planner_llm_user.txt', 'prompts/planning/analysis_path_llm_user.txt']:
    t = Path(f).read_text(encoding='utf-8')
    missing = find_missing_vars(t, ctx)
    print(f"  {f}: missing vars = {missing}")

print("\n" + "=" * 60)
print("3. 检查 RAG 检索（基于算法黄页）")
print("=" * 60)
r = retrieve("缺失值处理算法 均值填充 K邻近填充")
print(f"Query: '缺失值处理算法 均值填充 K邻近填充'")
print(f"返回 {len(r)} 条:")
for hit in r:
    print(f"  [{hit['_score']}] {hit['category_l1']} > {hit['category_l2']} > {hit['name']}")

r2 = retrieve("监督学习 分类算法 随机森林")
print(f"\nQuery: '监督学习 分类算法 随机森林'")
print(f"返回 {len(r2)} 条:")
for hit in r2:
    print(f"  [{hit['_score']}] {hit['category_l1']} > {hit['category_l2']} > {hit['name']}")

print("\n" + "=" * 60)
print("4. 检查 workflow_runner 工具函数")
print("=" * 60)
assert safe_object({"a": "", "b": 0}, None) == {"a": "", "b": 0}
assert safe_object({"a": ""}, {"a": "hi"}) == {"a": "hi"}
assert safe_object({"a": ""}, {"a": None}) == {"a": ""}
assert dig({"x": {"y": {"z": 1}}}, "x", "y", "z") == 1
assert dig({"x": {"y": {}}}, "x", "y", "z", default="d") == "d"
assert as_bool("true") == True
assert as_bool("false") == False
assert as_bool(1) == True
assert to_str(None) == ""
assert to_str("hi") == "hi"
assert to_str(123) == ""
print("✓ 辅助函数全部通过")

# parse_json_best_effort
p1 = parse_json_best_effort('{"a": 1}')
assert p1 == {"a": 1}
p2 = parse_json_best_effort('```json\n{"a":2}\n```')
assert p2 == {"a": 2}
p3 = parse_json_best_effort('一些前导文字\n{"a":3}\n后导')
assert p3 == {"a": 3}
p4 = parse_json_best_effort('not json at all')
assert p4.get("_parse_failed") == True
print("✓ parse_json_best_effort 鲁棒性测试通过")

print("\n" + "=" * 60)
print("5. 检查 plugins 模块")
print("=" * 60)
from workflows import _plugins
import pandas as pd

df = pd.DataFrame({'a':[1,2,3], 'b':[None,5,6]})
meta = _plugins.df_to_meta(df)
assert meta['is_success'] and meta['shape_0']==3 and meta['shape_1']==2
print(f"✓ df_to_meta: {meta['shape_0']}×{meta['shape_1']}")

s1 = _plugins.summary1_composer(desc="测试描述", head_dict_str='[{"a":1}]')
assert s1['summary_1']['title'] == '数据概览与数据含义分析'
print(f"✓ summary1_composer")

s2 = _plugins.summary2_composer(code="df.dropna()", desc="预处理说明", processed_df='[]')
assert s2['summary_2']['title'] == '数据预处理'
print(f"✓ summary2_composer")

s3 = _plugins.sec3_composer(fig_analysis=[{"fig":"{}", "analysis":"a1"}])
assert len(s3['summary_3']['fig_analysis']) == 1
print(f"✓ sec3_composer")

s4 = _plugins.sec4_composer(code="model.fit()", desc="建模", result="{}")
assert s4['summary_4']['title'] == '建模分析'
print(f"✓ sec4_composer")

fl = _plugins.final_list(
    processed_df_head_list=['h1','h2','final_head'],
    processed_df_list=['d1','d2','final_df'])
assert fl['processed_df_head'] == 'final_head'
assert fl['processed_df'] == 'final_df'
print(f"✓ final_list")

hc = _plugins.history_content_composer(content="new", history_content="old")
assert hc['history_content'] == "old\n\nnew"
print(f"✓ history_content_composer")

sc = _plugins.sec3_check_full(analysis_list=[{"analysis":"A"},{"analysis":"B"}])
assert sc['full'] == "A\n\nB"
print(f"✓ sec3_check_full")

pp = _plugins.get_preprocessing_suggestions(df=meta['df'])
assert pp['is_success'] and pp['n_rows'] == '3' and pp['missing_total'] == '1'
print(f"✓ get_preprocessing_suggestions: {pp['n_rows']}行/{pp['n_cols']}列/{pp['missing_total']}缺失")

fr = _plugins.format_recall(output_list=r)
assert "缺失值处理算法" in fr['knowledge_results']
print(f"✓ format_recall: {len(fr['knowledge_results'])} 字符")

# 本地 code_runner 执行测试（真的在子进程里跑代码）
cr = _plugins.code_runner(
    code="process_df = df.fillna(df.mean(numeric_only=True))",
    df=meta['df'])
assert cr['is_success'] == True, f"失败: {cr}"
print(f"✓ code_runner 成功跑通预处理代码")

# 错误代码测试
cr_err = _plugins.code_runner(code="raise ValueError('test')", df=meta['df'])
assert cr_err['is_success'] == False
print(f"✓ code_runner 错误捕获正常")

print("\n" + "=" * 60)
print("6. 检查 workflows 模块导入（不真调 LLM）")
print("=" * 60)
# 不能直接导入 workflows.planning 因为会触发 LLM 调用链，但能 import 模块本身
import importlib
import workflows.planning
import workflows.loading
print("✓ workflows.planning / workflows.loading 导入成功")
assert hasattr(workflows.planning, 'run_planning_workflow')
assert hasattr(workflows.loading, 'run_loading_workflow')
print("✓ 入口函数都在")

print("\n" + "=" * 60)
print("✅ 第②批所有静态/离线测试全部通过")
print("=" * 60)
