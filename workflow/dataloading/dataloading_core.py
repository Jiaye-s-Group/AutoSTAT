import csv
import io
import os
from typing import List, Optional

import chardet
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat, arff
import streamlit as st
import streamlit_antd_components as sac


def read_data_from_file(
    uploaded_data_file,
    col_names: Optional[List[str]] = None,
    sep: Optional[str] = None,
    na_values: List[str] = ['?'],
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    Read DataFrame from an uploaded data file.
    - Supports .csv/.data/.txt/.xlsx/.xls/.mat
    - When col_names=None, use header=0 (first row as column names)
    - When col_names is not None, use header=None and specify names=col_names
    - Text files: automatically detect encoding, sniff delimiter, skip bad lines
    - Excel files: directly use pandas.read_excel
    - MAT files: use scipy.loadmat, extract the first major variable, convert to DataFrame, and ensure 1D columns
    """
    # Read all bytes
    data_bytes = uploaded_data_file.read()
    # Reset stream position
    try:
        uploaded_data_file.seek(0)
    except Exception:
        pass

    name = uploaded_data_file.name
    ext = os.path.splitext(name)[1].lower()

    # Excel files handler
    if ext in ('.xlsx', '.xls'):
        excel_kwargs = {}
        if col_names is None:
            excel_kwargs['header'] = 0
        else:
            excel_kwargs['header'] = None
            excel_kwargs['names'] = col_names
        return pd.read_excel(io.BytesIO(data_bytes), **excel_kwargs)

    # Special handling for ARFF files
    if ext == '.arff':
        text = data_bytes.decode(encoding or 'utf-8', errors='ignore')
        raw_data, meta = arff.loadarff(io.StringIO(text))
        df = pd.DataFrame(raw_data)
        for col in df.select_dtypes([object]).columns:
            if isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8', errors='ignore')
        if col_names is not None and df.shape[1] == len(col_names):
            df.columns = col_names
        return df
        
    # —— Special handling for MAT files —— #
    if ext == '.mat':
        mat = loadmat(io.BytesIO(data_bytes))
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError('No valid data variable found in the MAT file.')
        arr = mat[data_keys[0]]

        # —— Handle sparse matrices first —— #
        if sparse.issparse(arr):
            arr = arr.toarray()

        arr = np.array(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        df = pd.DataFrame(arr)

        if col_names is not None and df.shape[1] == len(col_names):
            df.columns = col_names

        return df

    if encoding is None:
        detected = chardet.detect(data_bytes)
        encoding = detected.get('encoding', 'utf-8')
    sample = data_bytes[:10_000].decode(encoding, errors='ignore')

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        detected_sep = dialect.delimiter
        use_whitespace = False
    except csv.Error:
        detected_sep = None
        use_whitespace = True

    read_kwargs = {
        'engine': 'python',
        'encoding': encoding,
        'na_values': na_values,
        'comment': '|',
        'skipinitialspace': True,
        'on_bad_lines': 'skip',
    }
    if use_whitespace:
        read_kwargs['delim_whitespace'] = True
    else:
        read_kwargs['sep'] = detected_sep

    if col_names is None:
        read_kwargs['header'] = 0
    else:
        read_kwargs['header'] = None
        read_kwargs['names'] = col_names

    return pd.read_csv(io.BytesIO(data_bytes), **read_kwargs)


def process_complex_data(uploaded_files, dataloadingagent):
    """
    Upload processing logic:
    - Single file: Treated as a regular table or MAT file (first row as header)
    - Multiple files: If there is a .names/.arff header file, use its column names; otherwise infer column names
      And when multiple data files exist, perform horizontal or vertical concatenation based on user selection
    """
    if not uploaded_files:
        st.error("Please upload files first.")
        return None, None

    names_exts = ('.names', '.arff', '.doc')
    data_exts = ('.data', '.csv', '.txt', '.xlsx', '.xls', '.mat', '.arff', '.tsv', '.dat', '.tst')

    names_files = [f for f in uploaded_files
                   if os.path.splitext(f.name)[1].lower() in names_exts]
    data_files = [f for f in uploaded_files
                  if os.path.splitext(f.name)[1].lower() in data_exts]

    # Read directly when single file uploaded
    if len(uploaded_files) == 1 and uploaded_files[0] in data_files:
        return read_data_from_file(uploaded_files[0], col_names=None), None

    if not data_files:
        raise ValueError(
            "No data files detected, please upload supported formats: .csv/.data/.txt/.xlsx/.xls/.mat/.arff/.tsv/.dat/.tst"
        )

    # 1) If there is a header file (.names/.arff), read the column names
    if names_files:
        header_file = names_files[0]
        # Use read_data_from_file to read sample to ensure proper encoding handling
        sample_df = read_data_from_file(data_files[0], col_names=None)
        col_names = dataloadingagent.read_names_from_file(header_file, sample_df.head())
    else:
        # 2) Otherwise infer column names from the first data file, with encoding tolerance
        sample = data_files[0]
        ext0 = os.path.splitext(sample.name)[1].lower()
        try:
            if ext0 in ('.xlsx', '.xls'):
                col_names = list(pd.read_excel(sample, nrows=0))
            elif ext0 == '.mat':
                df_sample = read_data_from_file(sample, col_names=None)
                col_names = list(df_sample.columns)
            else:
                # Text file inference for column names, with encoding parameter
                # First detect with chardet, then try utf-8, fallback to latin1
                raw_bytes = sample.read()
                detected = chardet.detect(raw_bytes)
                enc = detected.get('encoding', 'utf-8')
                try:
                    col_names = list(pd.read_csv(
                        io.BytesIO(raw_bytes),
                        nrows=0,
                        encoding=enc,
                        engine='python'
                    ).columns)
                except UnicodeDecodeError:
                    col_names = list(pd.read_csv(
                        io.BytesIO(raw_bytes),
                        nrows=0,
                        encoding='latin1',
                        engine='python'
                    ).columns)
        finally:
            try: sample.seek(0)
            except: pass

    # Read all data files and unify column names
    dfs = [read_data_from_file(f, col_names=col_names) for f in data_files]

    # If multiple data files, pop up concatenation mode selection
    if len(data_files) >= 2:

        big_df = pd.concat(dfs, axis=0, ignore_index=True)

    else:
        big_df = dfs[0]

    return big_df, dfs


def load_from_path(local_path):

    ext = os.path.splitext(local_path)[1].lower()
    if ext in (".csv", ".txt", ".data"):
        df_local = pd.read_csv(local_path)
    elif ext in (".xls", ".xlsx"):
        df_local = pd.read_excel(local_path)
    elif ext == ".json":
        df_local = pd.read_json(local_path)
    elif ext == ".jsonl":
        df_local = pd.read_json(local_path, lines=True)
    elif ext == ".parquet":
        df_local = pd.read_parquet(local_path)
    elif ext in (".pkl", ".pickle"):
        df_local = pd.read_pickle(local_path)
    elif ext == ".feather":
        df_local = pd.read_feather(local_path)
    elif ext == ".arff":
        data, meta = arff.loadarff(local_path)
        df_local = pd.DataFrame(data)
        for col in df_local.select_dtypes([object]).columns:
            if isinstance(df_local[col].iloc[0], bytes):
                df_local[col] = df_local[col].str.decode('utf-8')
    else:
        st.error(f"Not supported file type: {ext}")
        df_local = None

    return df_local


def load_concat_file(dfs, agent):

    mode = sac.segmented(
        items=[
            sac.SegmentedItem(label='Vertical Concatenation'),
            sac.SegmentedItem(label='Horizontal Concatenation'),
        ], label='Multiple data files detected, please select concatenation method', size='sm', radius='sm'
    )

    if mode.startswith("Horizontal Concatenation"):
        dfs_pos = [df.reset_index(drop=True) for df in dfs]
        big_df = pd.concat(dfs_pos, axis=1)

        cols = []
        seen = {}
        for c in big_df.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        big_df.columns = cols
        agent.add_df(big_df)
    else:
        big_df = pd.concat(dfs, axis=0, ignore_index=True)
        agent.add_df(big_df)

    csv_bytes = big_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Download File",
    data=csv_bytes,
    file_name="processed_data.csv",
    mime="text/csv"
    )


class PathFileWrapper:
    """A wrapper to treat a local file path as a Streamlit UploadedFile."""
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self._file = None

    def read(self, *args, **kwargs):
        with open(self.path, 'rb') as f:
            return f.read()

    def seek(self, offset, whence=0):

        pass

    def __repr__(self):
        return f"PathFileWrapper(path='{self.path}')"