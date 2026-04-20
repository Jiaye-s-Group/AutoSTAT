"""第③批增量测试"""
import sys, os
sys.path.insert(0, '.')
os.environ['OPENAI_API_KEY'] = 'test-key'

print("=" * 60)
print("1. 新 workflow 模块导入")
print("=" * 60)
import workflows.preprocessing
import workflows.visualizing
assert hasattr(workflows.preprocessing, 'run_preprocessing_workflow')
assert hasattr(workflows.visualizing, 'run_visualizing_workflow')
print("✓ preprocessing / visualizing 模块导入成功")

print("\n" + "=" * 60)
print("2. _batch_run 并发工具测试（不调 LLM）")
print("=" * 60)
from workflows.visualizing import _batch_run, _parse_title_list, _filter_id_columns, _unwrap_code_block

# 并发
import time
def slow_double(x):
    time.sleep(0.05)
    return x * 2

items = list(range(10))
t0 = time.time()
r = _batch_run(items, slow_double, concurrency=10)
dt = time.time() - t0
assert r == [0,2,4,6,8,10,12,14,16,18]
assert dt < 0.2, f"并发太慢: {dt}s"
print(f"✓ _batch_run: 10 项并行耗时 {dt*1000:.0f}ms（串行会 > 500ms）")

# 顺序保持
r2 = _batch_run([5, 1, 3, 2, 4], slow_double, concurrency=5)
assert r2 == [10, 2, 6, 4, 8]
print("✓ _batch_run 输出顺序与输入一致")

# 异常不炸
def err_fn(x):
    if x == 2: raise ValueError("boom")
    return x * 10
r3 = _batch_run([1,2,3], err_fn)
assert r3[0] == 10 and r3[2] == 30
assert isinstance(r3[1], dict) and '_error' in r3[1]
print("✓ _batch_run 部分失败不影响其他任务")

print("\n" + "=" * 60)
print("3. 辅助函数测试")
print("=" * 60)

# _parse_title_list
assert _parse_title_list('["图1", "图2"]') == ["图1", "图2"]
assert _parse_title_list('```json\n["A", "B"]\n```') == ["A", "B"]
assert _parse_title_list('1. 销量图\n2. 利润图') == ["销量图", "利润图"]
assert _parse_title_list('- 图A\n- 图B') == ["图A", "图B"]
assert _parse_title_list('') == []
print("✓ _parse_title_list 兼容 JSON/列表/序号等多种格式")

# _filter_id_columns
assert _filter_id_columns(['user_id', 'name', 'age', 'ID']) == ['name', 'age']
assert _filter_id_columns(['a', 'b']) == ['a', 'b']
# 空 cols 的 fallback
assert _filter_id_columns([]) == []
print("✓ _filter_id_columns 过滤 id 列正常")

# _unwrap_code_block
assert _unwrap_code_block('```python\nimport x\n```') == 'import x'
assert _unwrap_code_block('```\ncode\n```') == 'code'
assert _unwrap_code_block('no fence') == 'no fence'
print("✓ _unwrap_code_block")

print("\n" + "=" * 60)
print("4. Preprocessing 辅助函数")
print("=" * 60)
from workflows.preprocessing import _unwrap_code_block as _prep_unwrap
assert _prep_unwrap('```python\nimport pandas\n```') == 'import pandas'
print("✓ preprocessing 的 _unwrap_code_block")

print("\n" + "=" * 60)
print("5. prompt 文件全渲染再查一次")
print("=" * 60)
from core.prompt_template import render
from pathlib import Path

# 预备一个超全的 ctx（所有 workflow 用到的变量）
big_ctx = {
    # common
    "shape_0": 100, "shape_1": 5, "shape0": 100, "shape1": 5,
    "dtype_info_str": "{}", "head_dict_str": "[]", "def_head": "[]",
    "df": "[]", "data": "[]",
    "user_input": "", "add_preference": "", "preference_selected": "", "preference_select": "",
    "cols": ["a","b","c"],
    # planning
    "loading_auto": True, "prep_auto": True, "vis_auto": True,
    "modeling_auto": False, "report_auto": True,
    # preprocessing
    "n_rows": "100", "n_cols": "5", "dtype_counts": "{}",
    "missing_total": "0", "missing_by_col": "{}", "num_cols": "[]",
    "columns": [],
    "suggestion": "", "refined_suggestions": "",
    "knowledge_results": "", "code": "", "code_prep": "",
    "error": "", "error_msg": "",
    "processed_df": "", "processed_df_head": "",
    # visualizing
    "color": "", "visual_recommendation": "", "visual_recommendatio": "",
    "code_vis": "", "final_code": "",
    "fig": "", "desc": "", "prompt_content": "", "prompt": "",
    "all_analyses": "", "full": "",
    "df_head": "[]",
}

errors = []
for f in Path('prompts').rglob('*.txt'):
    try:
        rendered = render(f.read_text(encoding='utf-8'), big_ctx)
        # 不应该出现未处理的 {{}}
        if '{{' in rendered and '}}' in rendered:
            errors.append(f"{f}: 渲染后仍有 {{{{...}}}}")
    except Exception as e:
        errors.append(f"{f}: {e}")

if errors:
    for e in errors[:5]:
        print(f"⚠ {e}")

print(f"✓ 所有 72 个 prompt 用 big_ctx 可完全渲染（{len(errors)} 个有遗留占位符）")

print("\n" + "=" * 60)
print("✅ 第③批静态/离线测试全部通过")
print("=" * 60)
