import csv
import hashlib
import io
import os
import re
from typing import List, Optional

import chardet
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat, arff
import streamlit as st
import streamlit_antd_components as sac

from utils.i18n import bt


def read_file_bytes(file_obj) -> bytes:
    if hasattr(file_obj, "getvalue"):
        data = file_obj.getvalue()
        return data if isinstance(data, bytes) else bytes(data)

    try:
        file_obj.seek(0)
    except Exception:
        pass
    data = file_obj.read()
    try:
        file_obj.seek(0)
    except Exception:
        pass
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def build_file_manifest(files) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for file_obj in files or []:
        data = read_file_bytes(file_obj)
        manifest.append(
            {
                "name": str(getattr(file_obj, "name", "")),
                "source": str(getattr(file_obj, "path", "") or ""),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return manifest


def file_manifest_fingerprint(manifest: list[dict[str, object]], *, combine_mode: str = "vertical") -> str:
    digest = hashlib.sha256()
    digest.update(combine_mode.encode("utf-8"))
    for item in manifest:
        digest.update(str(item.get("name", "")).encode("utf-8"))
        digest.update(str(item.get("source", "")).encode("utf-8"))
        digest.update(str(item.get("size", 0)).encode("ascii"))
        digest.update(str(item.get("sha256", "")).encode("ascii"))
    return digest.hexdigest()


def parse_names_file(header_file, expected_columns: int) -> list[str]:
    raw = read_file_bytes(header_file)
    encoding = chardet.detect(raw).get("encoding") or "utf-8"
    text = raw.decode(encoding, errors="replace")
    ext = os.path.splitext(str(getattr(header_file, "name", "")))[1].lower()

    if ext == ".arff":
        names = []
        for line in text.splitlines():
            match = re.match(r"^\s*@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", line, re.I)
            if match:
                names.append(next(value for value in match.groups() if value is not None))
    else:
        names = []
        for raw_line in text.splitlines():
            line = raw_line.split("|", 1)[0].strip()
            if not line or ":" not in line:
                continue
            name = line.split(":", 1)[0].strip().strip("'\"")
            if name:
                names.append(name)

    if len(names) == expected_columns:
        return names
    if len(names) + 1 == expected_columns:
        return [*names, "target"]
    raise ValueError(
        bt(
            f"表头文件解析得到 {len(names)} 个字段，但数据包含 {expected_columns} 列。",
            f"The header file defines {len(names)} fields, but the data contains {expected_columns} columns.",
        )
    )


def read_data_from_file(
    uploaded_data_file,
    col_names: Optional[List[str]] = None,
    sep: Optional[str] = None,
    na_values: List[str] = ['?'],
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    从上传的数据文件读取 DataFrame。
    - 支持 .csv/.data/.txt/.xlsx/.xls/.mat
    - col_names=None 时使用 header=0（文件首行做列名）
    - col_names 不为 None 时使用 header=None 并指定 names=col_names
    - 文本文件：自动探测编码、嗅探分隔符，跳过坏行
    - Excel 文件：直接使用 pandas.read_excel
    - MAT 文件：使用 scipy.loadmat，提取第一个主要变量，转为 DataFrame，并保证一维列
    """
    # 读取所有字节
    data_bytes = uploaded_data_file.read()
    # 重置流位置
    try:
        uploaded_data_file.seek(0)
    except Exception:
        pass

    name = uploaded_data_file.name
    ext = os.path.splitext(name)[1].lower()

    # Excel 文件处理
    if ext in ('.xlsx', '.xls'):
        excel_kwargs = {}
        if col_names is None:
            excel_kwargs['header'] = 0
        else:
            excel_kwargs['header'] = None
            excel_kwargs['names'] = col_names
        return pd.read_excel(io.BytesIO(data_bytes), **excel_kwargs)

    # ARFF 文件特殊处理
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
        
    # —— MAT 文件特殊处理 —— #
    if ext == '.mat':
        mat = loadmat(io.BytesIO(data_bytes))
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError(bt(
                "MAT 文件中未发现有效数据变量",
                "No valid data variable was found in the MAT file.",
            ))
        arr = mat[data_keys[0]]

        # —— 先处理稀疏矩阵 —— #
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
        det = chardet.detect(data_bytes)
        encoding = det.get("encoding", "utf-8")

    if encoding.lower() in ("utf-16", "utf-16le", "utf-16be", 
                            "utf-32", "utf-32le", "utf-32be"):
        text = data_bytes.decode(encoding, errors="ignore")
        data_bytes = text.encode("utf-8")
        encoding = "utf-8"

    sample = data_bytes[:10000].decode(encoding, errors="ignore")

    first_line = sample.splitlines()[0].strip()

    if sep is not None:
        detected_sep = sep
        use_whitespace = False

    elif "," in first_line:
        detected_sep = ","
        use_whitespace = False

    else:
        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=[",", ";", "\t", "|"]
            )
            detected_sep = dialect.delimiter
            use_whitespace = False
        except csv.Error:
            detected_sep = None
            use_whitespace = True  # fallback

    read_kwargs = {
        "engine": "python",
        "encoding": encoding,
        "na_values": na_values,
        "skipinitialspace": True,
        "on_bad_lines": "skip",
    }

    if col_names is None:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None
        read_kwargs["names"] = col_names

    if use_whitespace:
        read_kwargs["delim_whitespace"] = True
    else:
        read_kwargs["sep"] = detected_sep

    return pd.read_csv(io.BytesIO(data_bytes), **read_kwargs)


def process_complex_data(uploaded_files, dataloadingagent):
    """
    上传处理逻辑：
    - 单文件：当作普通表格或 MAT 文件读（第一行当表头）
    - 多文件：若有 .names/.arff 表头文件，则用其列名；否则推断列名
      并在存在多个数据文件时，通过用户选择进行横向或纵向拼接
    """
    if not uploaded_files:
        st.error(bt("请先上传文件", "Please upload files first."))
        return None, None

    names_exts = ('.names', '.arff', '.doc')
    data_exts = ('.data', '.csv', '.txt', '.xlsx', '.xls', '.mat', '.arff', '.tsv', '.dat', '.tst')

    names_files = [f for f in uploaded_files
                   if os.path.splitext(f.name)[1].lower() in names_exts]
    data_files = [f for f in uploaded_files
                  if os.path.splitext(f.name)[1].lower() in data_exts]

    # 单文件直接读取
    if len(uploaded_files) == 1 and uploaded_files[0] in data_files:
        return read_data_from_file(uploaded_files[0], col_names=None), None

    if not data_files:
        raise ValueError(
            bt(
                "未检测到任何数据文件，请上传支持的格式：.csv/.data/.txt/.xlsx/.xls/.mat/.arff/.tsv/.dat/.tst",
                "No data file was detected. Please upload a supported format: .csv/.data/.txt/.xlsx/.xls/.mat/.arff/.tsv/.dat/.tst",
            )
        )

    # 1) 如果存在表头文件 (.names/.arff)，读取列名
    if names_files:
        header_file = names_files[0]
        # 使用 read_data_from_file 读取 sample，以确保正确处理编码
        sample_df = read_data_from_file(data_files[0], col_names=None)
        col_names = parse_names_file(header_file, sample_df.shape[1])
    else:
        # 2) 否则从第一个数据文件推断列名，加入编码容错
        sample = data_files[0]
        ext0 = os.path.splitext(sample.name)[1].lower()
        try:
            if ext0 in ('.xlsx', '.xls'):
                col_names = list(pd.read_excel(sample, nrows=0))
            elif ext0 == '.mat':
                df_sample = read_data_from_file(sample, col_names=None)
                col_names = list(df_sample.columns)
            else:
                df_sample = read_data_from_file(sample, col_names=None)
                col_names = list(df_sample.columns)
        finally:
            try: sample.seek(0)
            except: pass

    # 外部表头文件意味着数据本身无表头；否则保留每个文件自己的首行表头，
    # 避免把 CSV 的表头再次读成第一行数据。
    if names_files:
        dfs = [read_data_from_file(f, col_names=col_names) for f in data_files]
    else:
        dfs = [read_data_from_file(f, col_names=None) for f in data_files]
        for item in dfs:
            if item.shape[1] != len(col_names):
                raise ValueError(
                    bt(
                        "多个数据文件的列数不一致，无法安全合并。",
                        "The uploaded data files have different column counts and cannot be combined safely.",
                    )
                )
            item.columns = col_names

    # 若多个数据文件，弹出拼接模式选择
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
        st.error(bt(f"不支持的文件类型：{ext}", f"Unsupported file type: {ext}"))
        df_local = None

    return df_local


def load_concat_file(dfs, agent):

    vertical_label = bt("纵向拼接", "Append Rows")
    horizontal_label = bt("横向拼接", "Join Columns")
    mode = sac.segmented(
        items=[
            sac.SegmentedItem(label=vertical_label),
            sac.SegmentedItem(label=horizontal_label),
        ], label=bt("检测到多个数据文件，请选择拼接方式", "Multiple data files detected. Choose how to combine them."), size='sm', radius='sm'
    )

    if mode == horizontal_label:
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
        combine_mode = "horizontal"
    else:
        big_df = pd.concat(dfs, axis=0, ignore_index=True)
        agent.add_df(big_df)
        combine_mode = "vertical"

    csv_bytes = big_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label=bt("下载文件", "Download File"),
    data=csv_bytes,
    file_name="processed_data.csv",
    mime="text/csv"
    )
    return big_df, combine_mode


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
