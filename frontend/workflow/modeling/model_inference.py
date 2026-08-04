import base64
import gzip
import io
import json
import traceback

import pandas as pd
import streamlit as st
from core.bounded_code_execution import (
    INFERENCE_TIMEOUT_SECONDS,
    run_bounded_safe_exec,
)

from utils.i18n import bt
from utils.sanitize_code import sanitize_code, to_json_serializable
from workflow.dataloading.dataloading_core import process_complex_data


def infer_load_data(agent) -> None:
    uploaded_files = st.file_uploader(
        bt("选择推理数据集", "Select Inference Dataset"),
        accept_multiple_files=True,
        help=bt("拖拽或点击以上传多个文件。", "Drag or click to upload multiple files."),
    )

    if uploaded_files:
        try:
            with st.spinner(bt("正在处理数据...", "Processing data...")):
                big_df, dfs = process_complex_data(uploaded_files, agent)
            if big_df is not None:
                agent.save_inference_data(big_df)
                st.success(bt("导入并处理完成！", "Import and processing complete."))
        except Exception as err:
            st.error(bt(f"导入失败：{err}", f"Import failed: {err}"))


def infer_execution(agent):
    inference_df = agent.load_inference_processed_df()
    edited_code = agent.load_inference_code()

    try:
        model_obj = agent.load_best_model()

        with st.spinner(bt("正在进行推断分析...", "Running inference analysis...")):
            execution_result = run_bounded_safe_exec(
                kind="inference",
                code=edited_code,
                dataframe=inference_df,
                timeout_seconds=INFERENCE_TIMEOUT_SECONDS,
                extra_values={"model_obj": model_obj},
            )
            if not execution_result["is_success"]:
                raise RuntimeError(str(execution_result["error"]))

            result_dict = execution_result.get("value")
            if result_dict is None:
                st.error(
                    bt(
                        "脚本未写入 `result_dict`。请确保编辑后的脚本在末尾赋值 `result_dict`。",
                        "The script did not write `result_dict`. Make sure the edited script assigns `result_dict` at the end.",
                    )
                )
            else:
                art = result_dict.get("artifacts", {})
                b64 = art.pop("predictions_df_b64", None)
                if not art:
                    result_dict.pop("artifacts", None)

                serializable = to_json_serializable(result_dict)
                try:
                    result_json = json.dumps(serializable, ensure_ascii=False)
                except Exception:
                    result_json = json.dumps(serializable, default=str, ensure_ascii=False)

                with st.expander(bt("推理结果", "Inference Results"), True):
                    if b64:
                        try:
                            gz_bytes = base64.b64decode(b64)
                            csv_bytes = gzip.decompress(gz_bytes)

                            df_pred = pd.read_csv(io.BytesIO(csv_bytes))
                            st.success(bt("已加载带预测结果的 DataFrame", "Loaded the DataFrame with predictions."))
                            st.dataframe(df_pred)

                            st.download_button(
                                label=bt("下载带预测结果（predictions.csv）", "Download Predictions (predictions.csv)"),
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(bt(f"解码 predictions_df 失败: {e}", f"Failed to decode predictions_df: {e}"))
                            records = result_dict.get("predictions_df_records")
                            if records:
                                try:
                                    df_pred = pd.DataFrame(records)
                                    st.dataframe(df_pred)
                                except Exception as e2:
                                    st.error(
                                        bt(
                                            f"从 records 恢复表格失败: {e2}",
                                            f"Failed to restore the table from records: {e2}",
                                        )
                                    )

    except Exception as e:
        st.error(bt(f"推断失败：{e}", f"Inference failed: {e}"))
        st.text(traceback.format_exc())
        raw = agent.code_generation_for_inference(agent.load_code(), inference_df.head())
        code = sanitize_code(raw)
        agent.save_inference_code(code)
