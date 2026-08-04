import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
import warnings
from utils.runtime_status import ensure_runtime_session_defaults, handle_runtime_callback
from utils.i18n import (
    UI_LANGUAGE_SESSION_KEY,
    bt,
    get_language,
    set_language,
    sync_report_language,
    t,
)
from utils.page_paths import asset_file, page_file
from utils.resizable_cards import inject_resizable_card_resizer
from settings.llm_config import render_llm_config_panel

from utils.workflow_state import (
    current_dataset_fingerprint,
    invalidate_from,
    record_stage_status,
    stable_fingerprint,
)
from core.planning_contract import DEFAULT_STAGE_PLAN
from core.plotly_serialization import figure_to_json, json_safe_figure
from core.figure_artifacts import normalize_figure_artifact


st.set_page_config(
    page_title="Autostat",
    page_icon="🤖",
    layout="wide",
)


# 忽略警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

def inject_sidebar_default_width() -> None:
    components.html(
        """
        <script>
        (() => {
          const parentWindow = window.parent;
          const doc = parentWindow.document;
          const STORAGE_KEY = "autostat:sidebar-user-sized:v1";
          const MIN_VIEWPORT_WIDTH = 1000;

          function collapseHostElement() {
            const frame = window.frameElement;
            if (!frame) {
              return;
            }

            const host = frame.closest('[data-testid="stElementContainer"]');
            [frame, host].filter(Boolean).forEach((element) => {
              element.style.setProperty("display", "block", "important");
              element.style.setProperty("height", "0", "important");
              element.style.setProperty("min-height", "0", "important");
              element.style.setProperty("max-height", "0", "important");
              element.style.setProperty("margin", "0", "important");
              element.style.setProperty("padding", "0", "important");
              element.style.setProperty("border", "0", "important");
              element.style.setProperty("overflow", "hidden", "important");
              element.style.setProperty("pointer-events", "none", "important");
            });
          }

          function desiredSidebarWidth() {
            return Math.min(310, Math.max(292, Math.round(parentWindow.innerWidth * 0.16)));
          }

          function getSidebar() {
            return doc.querySelector('section[data-testid="stSidebar"][aria-expanded="true"]');
          }

          function applyDefaultWidth() {
            if (parentWindow.sessionStorage.getItem(STORAGE_KEY) === "1") {
              return;
            }
            if (parentWindow.innerWidth < MIN_VIEWPORT_WIDTH) {
              return;
            }

            const sidebar = getSidebar();
            if (!sidebar) {
              return;
            }

            const targetWidth = desiredSidebarWidth();
            const currentWidth = sidebar.getBoundingClientRect().width;
            if (currentWidth > 0 && Math.abs(currentWidth - targetWidth) > 8) {
              sidebar.style.setProperty("width", `${targetWidth}px`);
              parentWindow.dispatchEvent(new Event("resize"));
            }
          }

          function rememberUserResize(event) {
            const sidebar = getSidebar();
            if (!sidebar) {
              return;
            }
            const rect = sidebar.getBoundingClientRect();
            if (Math.abs(event.clientX - rect.right) <= 18) {
              parentWindow.sessionStorage.setItem(STORAGE_KEY, "1");
            }
          }

          if (!parentWindow.__autostatSidebarDefaultWidthInstalled) {
            parentWindow.__autostatSidebarDefaultWidthInstalled = true;
            doc.addEventListener("pointerdown", rememberUserResize, true);
          }

          collapseHostElement();
          [50, 150, 350, 700, 1200].forEach((delay) => {
            parentWindow.setTimeout(applyDefaultWidth, delay);
          });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


class BaseAgent:
    def __init__(self):
        self.df = None
        self.memory = []
        self.code = None
        self.processed_df = None
        self.error = None
        self.finish_auto_task = False
    
    def load_df(self):
        return self.df
    
    def add_df(self, df):
        self.df = df
    
    def load_memory(self):
        return self.memory
    
    def add_memory(self, entry):
        self.memory.append(entry)
    
    def load_code(self):
        return self.code

    def save_code(self, code):
        self.code = code
    
    def load_processed_df(self):
        return self.processed_df

    def save_processed_df(self, processed_df):
        self.processed_df = processed_df
    
    def clear_memory(self):
        self.memory = []

    def save_error(self, error):
        self.error = error

    def save_inference_error(self, error):
        self.error = error

    def load_error(self):
        return self.error
    
    def finish_auto(self):
        self.finish_auto_task = True

class DataLoadingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.file_names = []
        self.dfs = []
        self.loading_workflow_result = None
    
    def load_file_name(self):
        return self.file_names
    
    def save_file_name(self, name):
        self.file_names.append(name)

    def replace_file_names(self, names):
        self.file_names = list(names or [])
    
    def save_dfs(self, dfs):
        self.dfs = dfs
    
    def load_dfs(self):
        return self.dfs

    def save_loading_workflow_result(self, loading_workflow_result):
        self.loading_workflow_result = loading_workflow_result

    def load_loading_workflow_result(self):
        return self.loading_workflow_result
    
    def read_names_from_file(self, header_file, sample_df):
        from workflow.dataloading.dataloading_core import parse_names_file

        return parse_names_file(header_file, sample_df.shape[1])
    
    def do_data_description(self, df, user_input):
        return bt(f"这是对'{user_input}'的响应", f"Response to '{user_input}'")

class PlannerAgent(BaseAgent):
    STAGE_ORDER = (
        "loading_auto",
        "prep_auto",
        "vis_auto",
        "modeling_auto",
        "report_auto",
    )
    STAGE_PAGES = {
        "loading_auto": ("dataloading", "dataloading_render.py"),
        "prep_auto": ("preprocessing", "preprocessing_render.py"),
        "vis_auto": ("visualization", "viz_render.py"),
        "modeling_auto": ("modeling", "modeling_render.py"),
        "report_auto": ("report", "report_render.py"),
    }
    STAGE_DEFAULTS = dict(DEFAULT_STAGE_PLAN)

    def __init__(self):
        super().__init__()
        self.loading_auto = True
        self.prep_auto = False
        self.switched_prep = False
        self.vis_auto = False
        self.switched_vis = False
        self.modeling_auto = False
        self.switched_modeling = False
        self.report_auto = False
        self.plan = None
        self.planning_result = None
        self.auto_plan = {stage: False for stage in self.STAGE_ORDER}

    def ensure_auto_plan_defaults(self):
        if not isinstance(getattr(self, "auto_plan", None), dict):
            self.auto_plan = {
                stage: bool(getattr(self, stage, False))
                for stage in self.STAGE_ORDER
            }
        for stage in self.STAGE_ORDER:
            self.auto_plan.setdefault(stage, bool(getattr(self, stage, False)))
        if not hasattr(self, "planning_result"):
            self.planning_result = None

    def _as_bool(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    def apply_plan(self, planning_result):
        self.ensure_auto_plan_defaults()
        planning_result = dict(planning_result or {})
        self.planning_result = planning_result
        self.plan = planning_result.get("plan") or ""
        self.switched_prep = False
        self.switched_vis = False
        self.switched_modeling = False
        self.auto_plan = {stage: False for stage in self.STAGE_ORDER}

        for stage in self.STAGE_ORDER:
            enabled = self._as_bool(
                planning_result.get(stage),
                default=self.STAGE_DEFAULTS.get(stage, False),
            )
            setattr(self, stage, enabled)
            self.auto_plan[stage] = enabled

    def stage_was_planned(self, stage):
        self.ensure_auto_plan_defaults()
        return bool(self.auto_plan.get(stage, False))

    def current_stage(self):
        for stage in self.STAGE_ORDER:
            if bool(getattr(self, stage, False)):
                return stage
        return None

    def current_page(self):
        stage = self.current_stage()
        if not stage:
            return None
        page_args = self.STAGE_PAGES.get(stage)
        if not page_args:
            return None
        return page_file(*page_args)

    def _finish_stage(self, stage):
        setattr(self, stage, False)
        return self.current_page()

    def self_driving(self, df):
        self.loading_auto = True
        self.prep_auto = False
        self.vis_auto = False
        self.modeling_auto = False
        self.report_auto = False
        self.switched_prep = False
        self.switched_vis = False
        self.switched_modeling = False
        self.auto_plan = {
            "loading_auto": True,
            "prep_auto": True,
            "vis_auto": True,
            "modeling_auto": True,
            "report_auto": True,
        }
        self.plan = bt(
            "自动模式已启动，将执行完整的数据分析流程",
            "Auto mode has started and will run the complete data analysis flow.",
        )
    
    def finish_loading_auto(self):
        self.switched_prep = True
        return self._finish_stage("loading_auto")
    
    def finish_prep_auto(self):
        self.switched_vis = True
        return self._finish_stage("prep_auto")
    
    def finish_vis_auto(self):
        self.switched_modeling = True
        return self._finish_stage("vis_auto")
    
    def finish_modeling_auto(self):
        return self._finish_stage("modeling_auto")

    def finish_report_auto(self):
        return self._finish_stage("report_auto")

    def stop_auto(self):
        self.loading_auto = False
        self.prep_auto = False
        self.vis_auto = False
        self.modeling_auto = False
        self.report_auto = False
        self.switched_prep = False
        self.switched_vis = False
        self.switched_modeling = False
        self.auto_plan = {stage: False for stage in self.STAGE_ORDER}

class DataPreprocessAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.preprocessing_suggestions = None
        self.user_input = None
        self.error = None
    
    def get_preprocessing_suggestions(self, user_input=None):
        return bt("这是预处理建议", "Here are preprocessing suggestions.")
    
    def save_preprocessing_suggestions(self, suggestions):
        self.preprocessing_suggestions = suggestions
    
    def save_user_input(self, user_input):
        self.user_input = user_input
    
    def refine_suggestions(self, df_head):
        pass
    
    def save_error(self, error):
        self.error = error
    
    def code_generation(self, df_head, suggest):
        return bt(
            "# 预处理代码示例\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n\n# 复制数据\nprocess_df = df.copy()\n\n# 标准化数值特征\nnumeric_cols = process_df.select_dtypes(include=['int64', 'float64']).columns\nscaler = StandardScaler()\nprocess_df[numeric_cols] = scaler.fit_transform(process_df[numeric_cols])\n\n# 处理缺失值\nprocess_df = process_df.fillna(process_df.mean())",
            "# Preprocessing code example\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n\n# Copy data\nprocess_df = df.copy()\n\n# Standardize numeric features\nnumeric_cols = process_df.select_dtypes(include=['int64', 'float64']).columns\nscaler = StandardScaler()\nprocess_df[numeric_cols] = scaler.fit_transform(process_df[numeric_cols])\n\n# Handle missing values\nprocess_df = process_df.fillna(process_df.mean())",
        )
    
    def load_preprocessing_suggestions(self):
        return self.preprocessing_suggestions

class VisualizationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.visualization_suggestions = None
        self.fig_desc_list = []
        self.suggestion = None
        self.user_input = None
        self.color = None
        self.error = None
    
    def get_visualization_suggestions(self):
        return bt("这是可视化建议", "Here are visualization suggestions.")
    
    def load_fig(self):
        return self.fig_desc_list
    
    def save_fig(self, fig_desc_list):
        self.fig_desc_list = fig_desc_list
    
    def save_suggestion(self, suggestion):
        self.suggestion = suggestion
    
    def load_suggestion(self):
        return self.suggestion
    
    def save_user_input(self, user_input):
        self.user_input = user_input
    
    def load_user_input(self):
        return self.user_input
    
    def save_color(self, color):
        self.color = color
    
    def load_color(self):
        return self.color

    def save_error(self, error):
        self.error = error

    def add_fig(
        self,
        fig,
        desc=None,
        base_fig=None,
        title=None,
        chart_id=None,
        fig_dict_key=None,
        generation_order=None,
        language=None,
    ):
        if base_fig is None:
            if hasattr(fig, "to_plotly_json"):
                try:
                    fig = json_safe_figure(fig)
                    base_fig = figure_to_json(fig)
                except Exception:
                    base_fig = fig
            else:
                base_fig = fig
        item = {"fig": fig, "base_fig": base_fig, "desc": desc}
        if title is not None:
            item["title"] = title
        if chart_id is not None:
            item["chart_id"] = chart_id
        if fig_dict_key is not None:
            item["fig_dict_key"] = fig_dict_key
        if generation_order is not None:
            item["generation_order"] = generation_order
        if language is not None:
            item["language"] = language
        self.fig_desc_list.append(
            normalize_figure_artifact(
                item,
                len(self.fig_desc_list),
                title=str(title or ""),
                description=str(desc or ""),
                language=str(language or ""),
            )
        )

    def code_generation(self, df_head, suggest):
        return ""

    def desc_fig(self, fig, dtype_info):
        return bt(
            f"图表已生成。字段类型概览: {dtype_info}",
            f"Chart generated. Field type overview: {dtype_info}",
        )

class ModelingCodingAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model_suggestions = None
        self.suggestion = None
        self.user_input = None
        self.target = None
        self.task_type = "auto"
        self.user_selection = None
        self.history_train_code = None
        self.modeling_result = None
        self.inference_data = None
        self.inference_processed_df = None
        self.inference_code = None
        self.best_model = None
        self.best_model_gz_bytes = None
        self.error = None
    
    def get_model_suggestions(self):
        return bt("这是建模建议", "Here are modeling suggestions.")
    
    def get_model_suggestion(self, user_input=None):
        return bt("这是建模建议", "Here are modeling suggestions.")
    
    def save_suggestion(self, suggestion):
        self.suggestion = suggestion
    
    def load_suggestion(self):
        return self.suggestion
    
    def save_user_input(self, user_input):
        self.user_input = user_input

    def load_user_input(self):
        return self.user_input

    def save_target(self, target):
        self.target = target

    def load_target(self):
        return self.target

    def save_task_type(self, task_type):
        self.task_type = task_type or "auto"

    def load_task_type(self):
        return self.task_type or "auto"

    def save_user_selection(self, user_selection):
        self.user_selection = user_selection

    def load_user_selection(self):
        return self.user_selection

    def save_history_train_code(self, history_train_code):
        self.history_train_code = history_train_code

    def load_history_train_code(self):
        return self.history_train_code

    def load_modeling_result(self):
        return self.modeling_result
    
    def save_modeling_result(self, modeling_result):
        self.modeling_result = modeling_result
    
    def load_inference_data(self):
        return self.inference_data
    
    def save_inference_data(self, inference_data):
        self.inference_data = inference_data
    
    def load_inference_processed_df(self):
        return self.inference_processed_df
    
    def save_inference_processed_df(self, inference_processed_df):
        self.inference_processed_df = inference_processed_df
    
    def load_inference_code(self):
        return self.inference_code
    
    def save_inference_code(self, inference_code):
        self.inference_code = inference_code
    
    def load_best_model(self):
        return self.best_model
    
    def save_best_model(self, best_model):
        self.best_model = best_model

    def load_best_model_gz_bytes(self):
        return self.best_model_gz_bytes

    def save_best_model_gz_bytes(self, best_model_gz_bytes):
        self.best_model_gz_bytes = best_model_gz_bytes

    def save_error(self, error):
        self.error = error

    def load_error(self):
        return self.error

    def result_format_prompt(self, result_json):
        return f"```json\n{result_json}\n```"
    
    def code_generation(self, df_head, selected_models):
        return bt('''# 建模代码示例
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设目标列是最后一列
target_col = df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
models = {}
models['Linear Regression'] = LinearRegression()
models['Random Forest'] = RandomForestRegressor()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.4f}")

# 保存最佳模型
best_model = min(models, key=lambda x: mean_squared_error(y_test, models[x].predict(X_test)))
print(f"最佳模型: {best_model}")''', '''# Modeling code example
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assume the target column is the last column
target_col = df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {}
models['Linear Regression'] = LinearRegression()
models['Random Forest'] = RandomForestRegressor()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.4f}")

# Save the best model
best_model = min(models, key=lambda x: mean_squared_error(y_test, models[x].predict(X_test)))
print(f"Best model: {best_model}")''')
    
    def code_generation_for_inference(self, code, inference_data_head):
        return bt(
            "# 推断代码示例\nimport pandas as pd\nimport numpy as np\n\n# 加载模型\n# 这里假设模型已经保存\n# model = joblib.load('best_model.joblib')\n\n# 进行预测\n# predictions = model.predict(inference_data)\n# print(predictions)",
            "# Inference code example\nimport pandas as pd\nimport numpy as np\n\n# Load model\n# This assumes the model has already been saved\n# model = joblib.load('best_model.joblib')\n\n# Run prediction\n# predictions = model.predict(inference_data)\n# print(predictions)",
        )
    
    def refine_suggestions(self):
        pass

class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.report_content = None
        self.report = None
        self.report_workflow_result = None
        self.report_format = "Word"
        self.gen_mode = bt("并行", "Parallel")
        self.outline_length = bt("标准", "Standard")
        self.outline = None
        self.word = None
        self.pdf = None
        self.pdf_export_method = None
        self.html = None
        self.markdown = None
        self.user_input = None
        self.report_language = "zh"
        self.report_current_language = "zh"
        self.report_language_versions = {}

    def ensure_report_language_defaults(self):
        if not getattr(self, "report_language", None):
            self.report_language = "zh"
        if not getattr(self, "report_current_language", None):
            self.report_current_language = self.report_language
        if not isinstance(getattr(self, "report_language_versions", None), dict):
            self.report_language_versions = {}
    
    def generate_report(self):
        return bt("这是报告内容", "This is the report content.")
    
    def load_report_format(self):
        return self.report_format
    
    def save_report_format(self, report_format):
        self.report_format = report_format
    
    def load_gen_mode(self):
        return self.gen_mode
    
    def save_gen_mode(self, gen_mode):
        self.gen_mode = gen_mode
    
    def load_outline_length(self):
        return self.outline_length
    
    def save_outline_length(self, outline_length):
        self.outline_length = outline_length
    
    def load_outline(self):
        return self.outline
    
    def save_outline(self, outline):
        self.outline = outline
    
    def load_word(self):
        return self.word
    
    def save_word(self, word):
        self.word = word

    def load_pdf(self):
        return self.pdf

    def save_pdf(self, pdf):
        self.pdf = pdf

    def load_pdf_export_method(self):
        return self.pdf_export_method

    def save_pdf_export_method(self, pdf_export_method):
        self.pdf_export_method = pdf_export_method
    
    def load_html(self):
        return self.html
    
    def save_html(self, html):
        self.html = html
    
    def load_markdown(self):
        return self.markdown
    
    def save_markdown(self, markdown):
        self.markdown = markdown
    
    def save_user_input(self, user_input):
        self.user_input = user_input

    def load_user_input(self):
        return self.user_input

    def load_report_language(self):
        self.ensure_report_language_defaults()
        return self.report_language

    def save_report_language(self, report_language):
        self.report_language = report_language or "zh"

    def load_report_current_language(self):
        self.ensure_report_language_defaults()
        return self.report_current_language

    def save_report_current_language(self, report_current_language):
        self.report_current_language = report_current_language or "zh"

    def load_report_language_versions(self):
        self.ensure_report_language_defaults()
        return self.report_language_versions

    def save_report_language_versions(self, report_language_versions):
        self.report_language_versions = (
            report_language_versions if isinstance(report_language_versions, dict) else {}
        )

    def load_report_language_version(self, report_language):
        self.ensure_report_language_defaults()
        return self.report_language_versions.get(report_language or "zh")

    def save_report_language_version(self, report_language, payload):
        self.ensure_report_language_defaults()
        if not isinstance(payload, dict):
            return
        self.report_language_versions[report_language or "zh"] = payload

    def load_report(self):
        return self.report

    def save_report(self, report):
        self.report = report

    def load_report_content(self):
        return self.report_content

    def save_report_content(self, report_content):
        self.report_content = report_content

    def load_report_workflow_result(self):
        return self.report_workflow_result

    def save_report_workflow_result(self, report_workflow_result):
        self.report_workflow_result = report_workflow_result
    
    def generate_toc_from_summary(self, summaries):
        return bt(
            "# 报告目录\n\n## 1. 数据导入\n## 2. 数据预处理\n## 3. 数据可视化\n## 4. 建模分析\n## 5. 结论与建议",
            "# Report Outline\n\n## 1. Data Import\n## 2. Data Preprocessing\n## 3. Data Visualization\n## 4. Modeling Analysis\n## 5. Conclusions and Recommendations",
        )
    
    def summary_html(self):
        return bt("数据可视化摘要", "Data visualization summary")
    
    def summary_word(self):
        return bt("数据可视化摘要", "Data visualization summary")

class Retriever:
    def __init__(self):
        self.learned_docs = []
        self._ref_retriever = None
        self._chunks = []
        self.last_error = ""
        self.last_results = []

    def _publish_index(self, chunks, retriever):
        self._chunks = list(chunks)
        self._ref_retriever = retriever
        st.session_state.ref_chunks = self._chunks
        st.session_state.ref_retriever = self._ref_retriever

    def add_uploaded_files_detailed(self, files):
        """逐文件解析并原子替换对应来源的检索索引。"""
        from core.ref_doc_parser import parse_and_chunk_results
        from core.ref_doc_retriever import RefDocRetriever

        parsed_results = parse_and_chunk_results(list(files or []))
        public_results = []
        for parsed in parsed_results:
            name = str(parsed.get("name") or "")
            status = str(parsed.get("status") or "failed")
            error = str(parsed.get("error") or "")
            chunk_count = int(parsed.get("chunk_count") or 0)

            if status == "success":
                candidate_chunks = [
                    chunk
                    for chunk in self._chunks
                    if str(chunk.get("source", "")) != name
                ]
                candidate_chunks.extend(parsed.get("chunks") or [])
                try:
                    candidate_retriever = RefDocRetriever(candidate_chunks)
                    self._publish_index(candidate_chunks, candidate_retriever)
                    if name not in self.learned_docs:
                        self.learned_docs.append(name)
                except Exception as exc:
                    status = "failed"
                    error = f"Index rebuild failed: {exc}"
                    chunk_count = 0

            public_results.append({
                "name": name,
                "status": status,
                "chunk_count": chunk_count,
                "error": error,
            })

        self.last_results = public_results
        failures = [result for result in public_results if result["status"] != "success"]
        self.last_error = "; ".join(
            f"{result['name']}: {result['error']}" for result in failures
        )
        return public_results
    
    def add_uploaded_files(self, files):
        """解析上传的参考资料文件，构建检索索引。"""
        try:
            results = self.add_uploaded_files_detailed(files)
            return sum(
                int(result.get("chunk_count") or 0)
                for result in results
                if result.get("status") == "success"
            )
        except Exception as e:
            self.last_error = str(e)
            return 0

    def remove_document(self, name):
        """删除单个来源；索引重建失败时保留旧索引。"""
        from core.ref_doc_retriever import RefDocRetriever

        source_name = str(name or "")
        candidate_chunks = [
            chunk
            for chunk in self._chunks
            if str(chunk.get("source", "")) != source_name
        ]
        try:
            candidate_retriever = RefDocRetriever(candidate_chunks) if candidate_chunks else None
            self._publish_index(candidate_chunks, candidate_retriever)
            self.learned_docs = [item for item in self.learned_docs if item != source_name]
            self.last_error = ""
            return True, ""
        except Exception as exc:
            self.last_error = str(exc)
            return False, str(exc)

    def rebuild_index(self):
        """使用已成功解析的 chunks 重建索引。"""
        from core.ref_doc_retriever import RefDocRetriever

        try:
            retriever = RefDocRetriever(self._chunks) if self._chunks else None
            self._publish_index(self._chunks, retriever)
            self.last_error = ""
            return True, ""
        except Exception as exc:
            self.last_error = str(exc)
            return False, str(exc)

    @property
    def ref_retriever(self):
        return self._ref_retriever

    @property
    def ref_chunks(self):
        return self._chunks

def init_session_state():
    """初始化会话状态，移除复杂的本地 API 配置逻辑"""
    
    from workflows.visualizing import (
        FIGURE_IMAGE_RENDER_STATE_SESSION_KEY,
        bind_figure_image_render_state,
    )

    render_state = st.session_state.get(FIGURE_IMAGE_RENDER_STATE_SESSION_KEY)
    if not isinstance(render_state, dict):
        render_state = {"disabled": False}
        st.session_state[FIGURE_IMAGE_RENDER_STATE_SESSION_KEY] = render_state
    bind_figure_image_render_state(render_state)

    if 'auto_mode' not in st.session_state:
        st.session_state.auto_mode = False

    if 'auto_planning' not in st.session_state:
        st.session_state.auto_planning = False

    if 'auto_planning_pending' not in st.session_state:
        st.session_state.auto_planning_pending = False
    
    # 初始化各个agent
    if 'data_loading_agent' not in st.session_state:
        st.session_state.data_loading_agent = DataLoadingAgent()
    elif not hasattr(st.session_state.data_loading_agent, "load_loading_workflow_result"):
        old_agent = st.session_state.data_loading_agent
        new_agent = DataLoadingAgent()
        if hasattr(old_agent, "__dict__"):
            new_agent.__dict__.update(old_agent.__dict__)
        st.session_state.data_loading_agent = new_agent
    
    if 'planner_agent' not in st.session_state:
        st.session_state.planner_agent = PlannerAgent()
    elif not hasattr(st.session_state.planner_agent, "apply_plan"):
        old_agent = st.session_state.planner_agent
        new_agent = PlannerAgent()
        if hasattr(old_agent, "__dict__"):
            new_agent.__dict__.update(old_agent.__dict__)
        new_agent.ensure_auto_plan_defaults()
        st.session_state.planner_agent = new_agent
    else:
        st.session_state.planner_agent.ensure_auto_plan_defaults()
    
    if 'data_preprocess_agent' not in st.session_state:
        st.session_state.data_preprocess_agent = DataPreprocessAgent()
    
    if 'visualization_agent' not in st.session_state:
        st.session_state.visualization_agent = VisualizationAgent()
    
    if 'modeling_coding_agent' not in st.session_state:
        st.session_state.modeling_coding_agent = ModelingCodingAgent()
    elif (
        not hasattr(st.session_state.modeling_coding_agent, "load_target")
        or not hasattr(st.session_state.modeling_coding_agent, "save_history_train_code")
        or not hasattr(st.session_state.modeling_coding_agent, "load_task_type")
    ):
        old_agent = st.session_state.modeling_coding_agent
        new_agent = ModelingCodingAgent()
        if hasattr(old_agent, "__dict__"):
            new_agent.__dict__.update(old_agent.__dict__)
        st.session_state.modeling_coding_agent = new_agent
    
    if 'report_agent' not in st.session_state:
        st.session_state.report_agent = ReportAgent()
    elif (
        not hasattr(st.session_state.report_agent, "load_report_content")
        or not hasattr(st.session_state.report_agent, "load_report_workflow_result")
        or not hasattr(st.session_state.report_agent, "load_user_input")
        or not hasattr(st.session_state.report_agent, "load_report_language")
    ):
        old_agent = st.session_state.report_agent
        new_agent = ReportAgent()
        if hasattr(old_agent, "__dict__"):
            new_agent.__dict__.update(old_agent.__dict__)
        new_agent.ensure_report_language_defaults()
        st.session_state.report_agent = new_agent
    else:
        st.session_state.report_agent.ensure_report_language_defaults()

    if UI_LANGUAGE_SESSION_KEY not in st.session_state:
        set_language(st.session_state.report_agent.load_report_language())
    else:
        set_language(get_language())
    sync_report_language(st.session_state.report_agent, get_language())
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = Retriever()
    
    if 'add_preference' not in st.session_state:
        st.session_state.add_preference = None
    
    if 'preference_selected' not in st.session_state:
        st.session_state.preference_selected = None

    if 'ref_chunks' not in st.session_state:
        st.session_state.ref_chunks = []

    if 'ref_retriever' not in st.session_state:
        st.session_state.ref_retriever = None

    ensure_runtime_session_defaults()


def run_app():
    """渲染 Streamlit 应用程序主入口"""
    init_session_state()
    handle_runtime_callback()
    inject_resizable_card_resizer()
    inject_sidebar_default_width()
    st.markdown(
        """
        <style>
        .stApp,
        .stApp * {
            font-family: "Times New Roman", "Microsoft YaHei", serif !important;
        }
        .stApp [data-testid="stIconMaterial"],
        .stApp .material-icons,
        .stApp .material-icons-outlined,
        .stApp .material-icons-round,
        .stApp .material-icons-sharp,
        .stApp .material-symbols-outlined,
        .stApp .material-symbols-rounded,
        .stApp .material-symbols-sharp,
        .stApp [class^="material-icons"],
        .stApp [class*=" material-icons"],
        .stApp [class^="material-symbols"],
        .stApp [class*=" material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-feature-settings: "liga" !important;
            -webkit-font-feature-settings: "liga" !important;
        }
        .stApp .bi,
        .stApp [class^="bi-"],
        .stApp [class*=" bi-"] {
            font-family: "bootstrap-icons" !important;
        }
        .stApp [data-testid="stAppViewContainer"],
        .stApp section[data-testid="stSidebar"],
        .stApp [data-testid="stVerticalBlock"],
        .stApp [data-testid="stHorizontalBlock"],
        .stApp [data-testid="stElementContainer"],
        .stApp [data-stale="true"],
        .stApp [class*="stale"],
        .stApp [class*="Stale"] {
            opacity: 1 !important;
            filter: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _reset_auto_agent_flags() -> None:
        for agent_key in (
            "data_loading_agent",
            "data_preprocess_agent",
            "visualization_agent",
            "modeling_coding_agent",
            "report_agent",
        ):
            agent = st.session_state.get(agent_key)
            if agent is not None:
                agent.finish_auto_task = False

    def _clear_auto_run_artifacts() -> None:
        for key in (
            "loading_workflow_result",
            "planning_workflow_result",
            "summary_1",
            "summary_2",
            "summary_3",
            "summary_4",
            "tu_title",
            "summary_1_title",
            "summary_1_desc",
            "summary_1_df",
            "abstract_1",
            "abstract_2",
            "abstract_3",
            "abstract_4",
            "suggestion",
            "prep_code_visible",
            "prep_result_from_summary_2",
            "viz_workflow_result",
            "viz_suggestion",
            "full",
            "visual_recommendatio",
            "final_code",
            "modeling_workflow_result",
            "modeling_suggestion",
            "model_suggestion",
            "modeling_summary_4",
            "modeling_abstract_4",
            "modeling_result_from_summary_4",
            "report_title",
            "report_workflow_outputs",
            "report_add_preference",
            "report_preference_selected",
            "report_selected_full_conten",
            "report_figure_ledger",
            "report_toc_text",
            "report_display_outline",
            "report_display_to_internal_toc_map",
            "report_load_abstract",
            "report_preproc_abstract",
            "report_visual_abstract",
            "report_coding_abstract",
            "report_final_html",
            "history_train_code_input",
            "history_train_code_reset_pending",
        ):
            st.session_state.pop(key, None)

        load_agent = st.session_state.get("data_loading_agent")
        if load_agent is not None:
            load_agent.clear_memory()
            load_agent.code = None
            load_agent.processed_df = None
            load_agent.loading_workflow_result = None

        preproc_agent = st.session_state.get("data_preprocess_agent")
        if preproc_agent is not None:
            preproc_agent.clear_memory()
            preproc_agent.code = None
            preproc_agent.processed_df = None
            preproc_agent.preprocessing_suggestions = None
            preproc_agent.user_input = None
            preproc_agent.error = None

        viz_agent = st.session_state.get("visualization_agent")
        if viz_agent is not None:
            viz_agent.clear_memory()
            viz_agent.code = None
            viz_agent.suggestion = None
            viz_agent.user_input = None
            viz_agent.error = None
            viz_agent.fig_desc_list = []

        modeling_agent = st.session_state.get("modeling_coding_agent")
        if modeling_agent is not None:
            modeling_agent.clear_memory()
            modeling_agent.code = None
            modeling_agent.suggestion = None
            modeling_agent.user_input = None
            modeling_agent.target = None
            modeling_agent.task_type = "auto"
            modeling_agent.user_selection = None
            modeling_agent.history_train_code = None
            modeling_agent.modeling_result = None
            modeling_agent.inference_data = None
            modeling_agent.inference_processed_df = None
            modeling_agent.inference_code = None
            modeling_agent.best_model = None
            modeling_agent.best_model_gz_bytes = None
            modeling_agent.error = None

        report_agent = st.session_state.get("report_agent")
        if report_agent is not None:
            report_agent.clear_memory()
            report_agent.report_content = None
            report_agent.report = None
            report_agent.report_workflow_result = None
            report_agent.outline = None
            report_agent.word = None
            report_agent.html = None
            report_agent.markdown = None
            report_agent.user_input = None
            report_agent.pdf = None
            report_agent.pdf_export_method = None
            report_agent.report_current_language = getattr(report_agent, "report_language", "zh")
            report_agent.report_language_versions = {}

    def _get_planning_ref_context(df) -> str:
        retriever = st.session_state.get("ref_retriever")
        if retriever is None:
            return ""
        is_empty = getattr(retriever, "is_empty", False)
        if callable(is_empty):
            try:
                is_empty = is_empty()
            except Exception:
                is_empty = False
        if is_empty:
            return ""
        try:
            columns = ", ".join(map(str, getattr(df, "columns", [])[:20]))
            query = bt(
                f"数据分析 业务背景 字段信息 {columns}",
                f"data analysis business context field information {columns}",
            )
            return retriever.retrieve_and_format(query, top_k=3)
        except Exception:
            return ""

    def _start_auto_mode() -> bool:
        df = st.session_state.data_loading_agent.load_df()
        if df is None:
            st.warning(t("auto.need_data"))
            st.session_state.auto_planning = False
            st.session_state.auto_planning_pending = False
            return False

        st.session_state.auto_planning = True
        invalidate_from(
            st.session_state,
            "planning",
            include_source=True,
            reason="automatic analysis replanned",
        )
        _clear_auto_run_artifacts()
        planner = st.session_state.planner_agent
        planner.stop_auto()

        try:
            from workflows.planning import run_planning_workflow

            planning_result = run_planning_workflow(
                df=df,
                add_preference=st.session_state.get("add_preference") or "",
                preference_selected=st.session_state.get("preference_selected") or "",
                ref_context=_get_planning_ref_context(df),
                language=get_language(),
            )

            planner.apply_plan(planning_result)
            st.session_state.planning_workflow_result = planning_result
            record_stage_status(
                st.session_state,
                "planning",
                "succeeded",
                input_fingerprint=current_dataset_fingerprint(st.session_state),
                output_fingerprint=stable_fingerprint(planning_result),
            )
            _reset_auto_agent_flags()

            if planner.current_page() is None:
                st.session_state.auto_mode = False
                st.info(t("auto.no_stage"))
                return False

            st.session_state.auto_mode = True
            return True
        except Exception as exc:
            st.session_state.auto_mode = False
            planner.stop_auto()
            st.error(t("auto.planning_failed", error=exc))
            return False
        finally:
            st.session_state.auto_planning = False
            st.session_state.auto_planning_pending = False

    def _queue_auto_mode_start() -> bool:
        if st.session_state.data_loading_agent.load_df() is None:
            st.warning(t("auto.need_data"))
            return False

        st.session_state.auto_mode = False
        st.session_state.auto_planning = True
        st.session_state.auto_planning_pending = True
        return True

    def _stop_auto_mode() -> None:
        st.session_state.auto_mode = False
        st.session_state.auto_planning = False
        st.session_state.auto_planning_pending = False
        st.session_state.planner_agent.stop_auto()
        _reset_auto_agent_flags()

    # --- 侧边栏布局 ---
    with st.sidebar:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] hr {
                margin-top: 0.14rem !important;
                margin-bottom: 0.14rem !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stExpander"] {
                margin-top: -0.24rem !important;
                margin-bottom: 0.22rem !important;
            }
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
                content: "AutoSTAT";
                display: block;
                margin: 0.06rem 0 0.28rem 0;
                color: #1e3a8a;
                font-weight: 700;
                font-size: 1.55rem;
            }
            .language-toggle-title {
                margin: 0.35rem 0 0.65rem 0.05rem;
                color: #111827;
                font-size: 1.04rem;
                font-weight: 400;
                letter-spacing: 0;
            }
            .st-key-ui_language_segmented_selector {
                width: 100%;
                margin-bottom: 0.72rem;
            }
            .st-key-ui_language_segmented_selector [data-testid="stButtonGroup"] {
                width: 100%;
                min-width: 0;
                max-width: 100%;
            }
            .st-key-ui_language_segmented_selector div[data-baseweb="button-group"] {
                display: flex;
                width: 100%;
                min-width: 0;
                max-width: 100%;
                background: linear-gradient(180deg, #fbfdff 0%, #f3f7ff 100%);
                border: 1px solid #d7e1f5;
                border-radius: 0.72rem;
                padding: 0.22rem;
                box-shadow:
                    inset 0 1px 2px rgba(255, 255, 255, 0.9),
                    0 10px 24px rgba(69, 107, 204, 0.13);
            }
            .st-key-ui_language_segmented_selector button {
                flex: 1 1 50% !important;
                width: 50% !important;
                min-width: 0 !important;
                min-height: 2.55rem;
                border: 0 !important;
                border-radius: 0.58rem !important;
                color: #253044 !important;
                background: transparent !important;
                font-size: 1rem !important;
                font-weight: 650 !important;
                transition:
                    color 0.18s ease,
                    background 0.18s ease,
                    box-shadow 0.18s ease,
                    transform 0.18s ease !important;
            }
            .st-key-ui_language_segmented_selector button:hover {
                color: #1d4ed8 !important;
                background: rgba(232, 240, 255, 0.62) !important;
            }
            .st-key-ui_language_segmented_selector button[aria-pressed="true"],
            .st-key-ui_language_segmented_selector button[aria-checked="true"],
            .st-key-ui_language_segmented_selector button[data-selected="true"],
            .st-key-ui_language_segmented_selector button[kind="segmented_controlActive"],
            .st-key-ui_language_segmented_selector button[data-testid="stBaseButton-segmented_controlActive"] {
                color: #1d4ed8 !important;
                background:
                    radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.95), rgba(241, 246, 255, 0.92) 58%, rgba(226, 237, 255, 0.88) 100%) !important;
                box-shadow:
                    0 7px 18px rgba(86, 122, 238, 0.22),
                    inset 0 1px 1px rgba(255, 255, 255, 0.96) !important;
                transform: translateY(-0.5px);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="language-toggle-title">{t("common.language")}</div>',
            unsafe_allow_html=True,
        )
        current_language = get_language()
        ui_language_widget_key = "ui_language_segmented_selector"
        ui_language_widget_sync_key = "ui_language_widget_synced"
        if (
            st.session_state.get(ui_language_widget_sync_key) != current_language
            and ui_language_widget_key in st.session_state
        ):
            st.session_state[ui_language_widget_key] = current_language
        st.session_state[ui_language_widget_sync_key] = current_language

        language_widget_kwargs = {}
        if ui_language_widget_key not in st.session_state:
            language_widget_kwargs["default"] = current_language
        selected_language = st.segmented_control(
            t("common.language"),
            options=["zh", "en"],
            format_func=lambda value: "中文" if value == "zh" else "English",
            key=ui_language_widget_key,
            label_visibility="collapsed",
            **language_widget_kwargs,
        )
        selected_language = selected_language or current_language
        if selected_language != current_language:
            set_language(selected_language)
            sync_report_language(st.session_state.get("report_agent"), selected_language)
            st.session_state[ui_language_widget_sync_key] = selected_language
            st.rerun()

        with st.expander(t("sidebar.llm_config"), expanded=True):
            render_llm_config_panel()

        # 清空数据按钮
        if st.button(t("sidebar.clear_all"), use_container_width=True):
            preserved_keys = {
                UI_LANGUAGE_SESSION_KEY,
                "user_id",
                "user_code",
                "quota_balance",
                "auth_mode",
                "auth_mode_radio",
                "llm_api_key",
                "llm_base_url",
                "llm_model",
                "llm_provider",
                "llm_key_input",
                "llm_url_input",
                "llm_model_input",
                "llm_configured",
                "llm_connection_signature",
            }
            preserved_state = {
                key: st.session_state[key]
                for key in preserved_keys
                if key in st.session_state
            }
            preserved_state[UI_LANGUAGE_SESSION_KEY] = get_language()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.update(preserved_state)
            st.rerun()

        # 自动模式逻辑（保留核心流程控制）
        df = st.session_state.data_loading_agent.load_df()
        auto_control_active = bool(
            st.session_state.auto_mode
            or st.session_state.auto_planning
            or st.session_state.auto_planning_pending
        )
        if not auto_control_active:
            if st.button(t("sidebar.start_auto"), use_container_width=True, type="primary"):
                if _queue_auto_mode_start():
                    st.rerun()
        else:
            if st.button(t("sidebar.stop_auto"), use_container_width=True):
                _stop_auto_mode()
                st.rerun()

        # 检查logo目录是否存在
        logo_path = asset_file("logo", "logo_big.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)

    # --- 页面导航 (保持模块化) ---
    pages = {
        t("app.nav.analysis_flow"): [
            st.Page(page_file("dataloading", "dataloading_render.py"), title=t("app.page.data_loading")),
            st.Page(page_file("preprocessing", "preprocessing_render.py"), title=t("app.page.preprocessing")),
            st.Page(page_file("visualization", "viz_render.py"), title=t("app.page.visualization")),
            st.Page(page_file("modeling", "modeling_render.py"), title=t("app.page.modeling")),
            st.Page(page_file("report", "report_render.py"), title=t("app.page.report")),
        ],
        t("app.nav.system_config"): [
            st.Page(page_file("preference", "pref_render.py"), title=t("app.page.preference")),
        ]
    }
    pg = st.navigation(pages, position="sidebar")

    if st.session_state.get("auto_planning_pending"):
        with st.spinner(
            bt(
                "正在规划自动分析流程，请耐心等待。",
                "Planning the automated analysis flow. Please wait.",
            )
        ):
            auto_mode_started = _start_auto_mode()
        if auto_mode_started:
            next_page = st.session_state.planner_agent.current_page()
            if next_page is not None:
                st.switch_page(next_page)

    pg.run()


if __name__ == "__main__":
    run_app()
