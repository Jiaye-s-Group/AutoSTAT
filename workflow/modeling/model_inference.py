import base64
import gzip
import io
import json
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

from workflow.dataloading.dataloading_core import process_complex_data
from utils.sanitize_code import sanitize_code, to_json_serializable


def infer_load_data(agent) -> None:

    uploaded_files = st.file_uploader(
        "选择推理数据集",
        accept_multiple_files=True,
        help="拖拽或点击上传多个文件",
    )

    if uploaded_files:
        try:
            with st.spinner("正在处理数据..."):
                big_df, dfs = process_complex_data(uploaded_files, agent)
            if big_df is not None:
                agent.save_inference_data(big_df)
                st.success("导入并处理完成！")
        except Exception as err:
            st.error(f"导入失败：{err}")


def infer_execution(agent):

    inference_df = agent.load_inference_processed_df()
    edited_code = agent.load_inference_code()

    try:
        model_obj = agent.load_best_model()
        
        exec_ns = {
            "inference_df": inference_df,
            'model_obj': model_obj,
            "np": np,
            "pd": pd,
            "StandardScaler": StandardScaler
        }
        
        with st.spinner("正在进行推断分析..."):
            exec(edited_code, exec_ns)
            
            result_dict = exec_ns.get("result_dict")
            if result_dict is None:
                st.error("脚本未写入 `result_dict`。请确保编辑后的脚本在末尾赋值 result_dict。")
            else:
                art = result_dict.get('artifacts', {})
                b64 = art.pop('predictions_df_b64', None)
                if not art:
                    result_dict.pop('artifacts', None)

                serializable = to_json_serializable(result_dict)
                try:
                    result_json = json.dumps(serializable, ensure_ascii=False)
                except Exception:
                    result_json = json.dumps(serializable, default=str, ensure_ascii=False)

                with st.expander("推理结果", True):
                    if b64:
                        try:
                            gz_bytes = base64.b64decode(b64)
                            csv_bytes = gzip.decompress(gz_bytes)

                            df_pred = pd.read_csv(io.BytesIO(csv_bytes))
                            st.success("已加载带预测结果的 DataFrame")
                            st.dataframe(df_pred)

                            st.download_button(
                                label="下载带预测结果（predictions.csv）",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"解码 predictions_df 失败: {e}")
                            # 兜底：尝试从 records 字段恢复
                            records = result_dict.get('predictions_df_records')
                            if records:
                                try:
                                    df_pred = pd.DataFrame(records)
                                    st.dataframe(df_pred)
                                except Exception as e2:
                                    st.error(f"从 records 恢复表格失败: {e2}")

    except Exception as e:
        st.error(f"推断失败：{e}")
        st.text(traceback.format_exc())
        agent.save_inference_error(traceback.format_exc())
        raw = agent.code_generation_for_inference(agent.load_code(), inference_data.head(), auto=True)
        code = sanitize_code(raw)
        agent.save_inference_code(code)