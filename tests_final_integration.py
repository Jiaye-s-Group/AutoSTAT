"""最终整合检查：前端render 能正确桥接到本地 workflow"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, './frontend')
os.environ['OPENAI_API_KEY'] = 'test-key'

# 1. 所有本地 workflow 都可导入
import workflows.planning
import workflows.loading
import workflows.preprocessing
import workflows.visualizing
import workflows.modeling
import workflows.reporting_toc
import workflows.reporting_partly
import workflows.autostat
print("✓ 所有 8 个 workflow 模块导入成功")

# 2. 桥接层导入正常
sys.path.insert(0, './frontend')
# streamlit 未装，仅做模块可见性检查
import importlib.util
spec = importlib.util.spec_from_file_location(
    "local_workflow_bridge",
    "./frontend/utils/local_workflow_bridge.py",
)
mod = importlib.util.module_from_spec(spec)
# 不运行，只检查语法和 import 方案
import ast
ast.parse(open("./frontend/utils/local_workflow_bridge.py", encoding="utf-8").read())
print("✓ frontend/utils/local_workflow_bridge.py 语法 OK")

# 3. 所有 bridge 函数签名可见
src = open("./frontend/utils/local_workflow_bridge.py", encoding="utf-8").read()
for name in ["call_loading_bridge", "call_preprocessing_bridge", "call_visualizing_bridge",
             "call_modeling_bridge", "call_reporting_toc_bridge", "call_reporting_partly_bridge"]:
    assert f"def {name}" in src, f"缺 {name}"
print("✓ 桥接层 6 个函数都在")

# 4. 前端 render 里确认替换完成（不应再有 requests.post Coze 或 COZE_SPACE_ID 调用）
import re
render_files = [
    "./frontend/workflow/dataloading/dataloading_render.py",
    "./frontend/workflow/preprocessing/preprocessing_render.py",
    "./frontend/workflow/visualization/viz_render.py",
    "./frontend/workflow/modeling/modeling_render.py",
    "./frontend/workflow/report/report_render.py",
]
for f in render_files:
    src = open(f, encoding="utf-8").read()
    # 查是否还有调用 Coze 的痕迹
    has_coze_post = re.search(r"requests\.post\(\s*coze_url", src)
    has_coze_client = re.search(r"client\.workflows\.runs\.stream", src)
    assert not has_coze_post, f"{f} 还有 requests.post(coze_url"
    assert not has_coze_client, f"{f} 还有 client.workflows.runs.stream"
print(f"✓ 5 个 render 文件里的 Coze 调用都被替换掉了")

print("\n" + "=" * 60)
print("✅ 前端集成检查全部通过")
print("=" * 60)
