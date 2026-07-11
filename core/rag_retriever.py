"""BM25 retriever for the built-in statistical-method catalog."""
from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import jieba

    if hasattr(jieba, "setLogLevel"):
        jieba.setLogLevel(logging.ERROR)
    logging.getLogger("jieba").setLevel(logging.ERROR)
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


DEFAULT_JSONL = os.getenv(
    "RAG_KNOWLEDGE_PATH",
    str(Path(__file__).resolve().parent.parent / "knowledge" / "algorithm_catalog.jsonl"),
)
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
DEFAULT_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese and Latin text for BM25 matching."""
    if not text:
        return []
    text = text.lower()
    if _HAS_JIEBA:
        tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    else:
        tokens = _TOKEN_PATTERN.findall(text)
    # Drop short Latin/numeric noise while preserving single CJK characters.
    return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]


class Retriever:
    def __init__(self, jsonl_path: str | Path = DEFAULT_JSONL):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {self.path}")

        self.docs: list[dict[str, Any]] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_field_tokens: list[dict[str, list[str]]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_len: float = 0.0
        self._N: int = 0

        self._load()
        self._build_index()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not self.docs:
            raise RuntimeError(f"知识库为空: {self.path}")

    def _build_index(self) -> None:
        for doc in self.docs:
            # Generated queries usually emphasize category_l2 and algorithm name.
            name = str(doc.get("name", ""))
            cat2 = str(doc.get("category_l2", ""))
            cat1 = str(doc.get("category_l1", ""))
            desc = str(doc.get("description", ""))

            field_tokens = {
                "name": _tokenize(name),
                "category_l2": _tokenize(cat2),
                "category_l1": _tokenize(cat1),
                "description": _tokenize(desc),
            }
            # Weight the most discriminative fields in the BM25 document body.
            all_tokens = (
                field_tokens["name"] * 3
                + field_tokens["category_l2"] * 3
                + field_tokens["category_l1"]
                + field_tokens["description"]
            )
            self._doc_tokens.append(all_tokens)
            self._doc_field_tokens.append(field_tokens)

            # document frequency
            seen: set[str] = set()
            for t in set(all_tokens):
                if t in seen:
                    continue
                seen.add(t)
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1

        self._N = len(self.docs)
        self._avg_len = sum(len(t) for t in self._doc_tokens) / max(1, self._N)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        k1, b = 1.5, 0.75
        scores: list[tuple[float, int]] = []
        for i, d_tokens in enumerate(self._doc_tokens):
            score = 0.0
            dl = len(d_tokens)
            # Accumulate BM25 contributions for each query token.
            for qt in q_tokens:
                df = self._doc_freq.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
                # Term frequency in this document.
                tf = d_tokens.count(qt)
                if tf == 0:
                    continue
                norm = 1 - b + b * (dl / max(1, self._avg_len))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * norm))
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Normalize scores by the top hit.
        if not scores:
            return []
        max_score = scores[0][0] or 1.0

        results: list[dict[str, Any]] = []
        for s, i in scores[:top_k]:
            norm_score = s / max_score
            if norm_score < min_score:
                break
            doc = dict(self.docs[i])
            doc["_score"] = round(norm_score, 4)
            results.append(doc)
        return results


# Singleton convenience API.

_instance: Retriever | None = None


def get_retriever() -> Retriever:
    global _instance
    if _instance is None:
        _instance = Retriever()
    return _instance


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Retrieve matching methods from the built-in catalog."""
    return get_retriever().retrieve(query, top_k=top_k, min_score=min_score)


def format_recall(results: list[dict[str, Any]]) -> str:
    """Format retrieved methods as Markdown for downstream prompts."""
    if not results:
        return "（未召回相关算法）"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r.get('name', '?')}")
        lines.append(f"- **分类**：{r.get('category_l1', '')} > {r.get('category_l2', '')}")
        desc = r.get("description", "")
        if desc:
            lines.append(f"- **简述**：{desc}")
        code = r.get("code", "")
        if code:
            lines.append("- **参考代码**：")
            lines.append("```python")
            lines.append(code.strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)
