"""
参考资料文档解析器。

支持 PDF、DOCX、TXT 三种格式。
解析后将文档内容按段落拆分为 chunk 列表，供 RefDocRetriever 检索。

公开 API：
    parse_and_chunk(files, chunk_size=800, overlap=100) -> list[dict]
"""
from __future__ import annotations

import io
import re
from typing import Any, BinaryIO


# ---- 文本提取 ----


def _extract_pdf_text(file_obj: BinaryIO) -> str:
    """从 PDF 文件提取文本。优先 PyMuPDF，降级 pdfplumber。"""
    raw = file_obj.read()
    # 尝试 PyMuPDF (fitz)
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=raw, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass

    # 降级 pdfplumber
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass

    return ""


def _extract_docx_text(file_obj: BinaryIO) -> str:
    """从 DOCX 文件提取文本。"""
    try:
        from docx import Document  # type: ignore

        doc = Document(file_obj)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        return ""
    except Exception:
        return ""


def _extract_txt_text(file_obj: BinaryIO) -> str:
    """从 TXT 文件提取文本。"""
    raw = file_obj.read()
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ---- 分块 ----


def _split_paragraphs(text: str) -> list[str]:
    """按空行 / 换行分段，过滤空段落。"""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _sliding_window_chunks(
    paragraphs: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """
    将段落列表合并后，按字符数滑窗切分。
    - chunk_size: 每个 chunk 的目标字符数
    - overlap: 相邻 chunk 的重叠字符数
    """
    if not paragraphs:
        return []

    full_text = "\n\n".join(paragraphs)
    if len(full_text) <= chunk_size:
        return [full_text]

    chunks: list[str] = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]

        # 尝试在自然分割点（句号、换行）截断
        if end < len(full_text):
            # 寻找最后一个自然分割点
            for sep in ["\n\n", "\n", "。", ".", "；", ";", "！", "!"]:
                last = chunk.rfind(sep)
                if last > chunk_size // 2:
                    chunk = chunk[: last + len(sep)]
                    end = start + len(chunk)
                    break

        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(full_text):
            break

    return [c for c in chunks if c]


# ---- 主函数 ----


def parse_and_chunk(
    files: list[Any],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    解析多个上传文件，返回 chunk 列表。

    Parameters
    ----------
    files : list
        Streamlit UploadedFile 对象列表（需具有 .name 和 .read() 方法）
    chunk_size : int
        每个 chunk 的目标字符数
    overlap : int
        相邻 chunk 的重叠字符数

    Returns
    -------
    list[dict]
        [{"text": "...", "source": "filename.pdf", "chunk_id": "filename.pdf::0"}, ...]
    """
    all_chunks: list[dict[str, Any]] = []

    for f in files:
        name = getattr(f, "name", str(f))
        # 重置文件指针
        if hasattr(f, "seek"):
            f.seek(0)

        lower_name = name.lower()
        if lower_name.endswith(".pdf"):
            text = _extract_pdf_text(f)
        elif lower_name.endswith(".docx"):
            text = _extract_docx_text(f)
        elif lower_name.endswith((".txt", ".names", ".md")):
            text = _extract_txt_text(f)
        else:
            # 尝试当 TXT 处理
            text = _extract_txt_text(f)

        if not text.strip():
            continue

        paragraphs = _split_paragraphs(text)
        chunks = _sliding_window_chunks(paragraphs, chunk_size, overlap)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "source": name,
                    "chunk_id": f"{name}::{i}",
                }
            )

    return all_chunks
