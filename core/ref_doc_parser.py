"""Parse uploaded reference documents into retrievable text chunks."""
from __future__ import annotations

import io
import re
from typing import Any, BinaryIO


# Text extraction.


def _extract_pdf_text(file_obj: BinaryIO) -> str:
    """Extract PDF text, preferring PyMuPDF with pdfplumber as fallback."""
    raw = file_obj.read()
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
    """Extract text from a DOCX file."""
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
    """Extract text from a plain-text file."""
    raw = file_obj.read()
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# Chunking.


def _split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _sliding_window_chunks(
    paragraphs: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Merge paragraphs and split them into overlapping character windows."""
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

        if end < len(full_text):
            # Prefer ending chunks at a sentence or paragraph boundary.
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


# Public parser.


def parse_and_chunk(
    files: list[Any],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """Parse uploaded files into chunks for reference-document retrieval."""
    all_chunks: list[dict[str, Any]] = []

    for f in files:
        name = getattr(f, "name", str(f))
        # Reset the file pointer before reading uploaded files.
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
            # Treat unknown file types as plain text when possible.
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
