"""
参考资料文档检索器 —— 基于用户上传文档的 BM25 检索。

与 rag_retriever.py 同构，但数据源是用户上传的 PDF/DOCX 文档 chunks，
而非静态的算法黄页 JSONL。

公开 API：
    RefDocRetriever(chunks)
        .retrieve(query, top_k=3, min_score=0.2) -> list[dict]
        .format_results(results) -> str
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

try:
    import jieba  # type: ignore

    if hasattr(jieba, "setLogLevel"):
        jieba.setLogLevel(logging.ERROR)
    logging.getLogger("jieba").setLevel(logging.ERROR)
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# ---- 分词（与 rag_retriever.py 保持一致） ----

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    if _HAS_JIEBA:
        tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    else:
        tokens = _TOKEN_PATTERN.findall(text)
    return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]


# ---- BM25 检索器 ----


class RefDocRetriever:
    """基于 BM25 的参考资料检索器。"""

    def __init__(self, chunks: list[dict[str, Any]]):
        """
        Parameters
        ----------
        chunks : list[dict]
            [{"text": "...", "source": "...", "chunk_id": "..."}, ...]
        """
        self.chunks = chunks or []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_len: float = 0.0
        self._N: int = 0
        if self.chunks:
            self._build_index()

    def _build_index(self) -> None:
        for chunk in self.chunks:
            text = str(chunk.get("text", ""))
            source = str(chunk.get("source", ""))
            # source 名也作为 tokens 加入（加权 2 倍），方便按文件名检索
            tokens = _tokenize(text) + _tokenize(source) * 2
            self._doc_tokens.append(tokens)

            for t in set(tokens):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        self._N = len(self.chunks)
        self._avg_len = sum(len(t) for t in self._doc_tokens) / max(1, self._N)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.2,
    ) -> list[dict[str, Any]]:
        """BM25 检索，返回 top_k 个最相关的 chunk。"""
        if not self.chunks or not query:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        k1, b = 1.5, 0.75
        scores: list[tuple[float, int]] = []

        for i, d_tokens in enumerate(self._doc_tokens):
            score = 0.0
            dl = len(d_tokens)
            for qt in q_tokens:
                df = self._doc_freq.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
                tf = d_tokens.count(qt)
                if tf == 0:
                    continue
                norm = 1 - b + b * (dl / max(1, self._avg_len))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * norm))
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        if not scores:
            return []
        max_score = scores[0][0] or 1.0

        results: list[dict[str, Any]] = []
        for s, i in scores[:top_k]:
            norm_score = s / max_score
            if norm_score < min_score:
                break
            doc = dict(self.chunks[i])
            doc["_score"] = round(norm_score, 4)
            results.append(doc)
        return results

    def format_results(self, results: list[dict[str, Any]]) -> str:
        """将检索结果格式化为 LLM prompt 可读的文本。"""
        if not results:
            return "（未上传参考资料或未召回相关内容）"
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "未知来源")
            text = r.get("text", "")
            score = r.get("_score", 0)
            parts.append(
                f"### 参考文档片段 {i}（来源：{source}，相关度：{score:.2f}）\n{text}"
            )
        return "\n\n".join(parts)

    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.2,
    ) -> str:
        """一步到位：检索 + 格式化。"""
        results = self.retrieve(query, top_k=top_k, min_score=min_score)
        return self.format_results(results)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0
