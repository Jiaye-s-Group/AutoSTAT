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
        "Select inference dataset",
        accept_multiple_files=True,
        help="Drag and drop or click to upload multiple files",
    )

    if uploaded_files:
        try:
            with st.spinner("Processing data..."):
                big_df, dfs = process_complex_data(uploaded_files, agent)
            if big_df is not None:
                agent.save_inference_data(big_df)
                st.success("Import and processing completed!")
        except Exception as err:
            st.error(f"Import failed: {err}")


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
        
        with st.spinner("Performing inference analysis..."):
            exec(edited_code, exec_ns)
            
            result_dict = exec_ns.get("result_dict")
            if result_dict is None:
                st.error("The script did not write `result_dict`. Please ensure the edited script assigns result_dict at the end.")
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

                with st.expander("Inference Results", True):
                    if b64:
                        try:
                            gz_bytes = base64.b64decode(b64)
                            csv_bytes = gzip.decompress(gz_bytes)

                            df_pred = pd.read_csv(io.BytesIO(csv_bytes))
                            st.success("Successfully loaded DataFrame with predictions")
                            st.dataframe(df_pred)

                            st.download_button(
                                label="Download predictions (predictions.csv)",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Failed to decode predictions_df: {e}")
                            # Fallback: attempt to recover from records field
                            records = result_dict.get('predictions_df_records')
                            if records:
                                try:
                                    df_pred = pd.DataFrame(records)
                                    st.dataframe(df_pred)
                                except Exception as e2:
                                    st.error(f"Failed to restore table from records: {e2}")

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.text(traceback.format_exc())
        agent.save_inference_error(traceback.format_exc())
        raw = agent.code_generation_for_inference(agent.load_code(), inference_data.head(), auto=True)
        code = sanitize_code(raw)
        agent.save_inference_code(code)