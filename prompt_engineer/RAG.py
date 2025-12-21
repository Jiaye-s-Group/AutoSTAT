import os
import faiss
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from sentence_transformers import SentenceTransformer
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import io


@st.cache_resource # 关键：让模型和索引在全局只加载一次
def get_retriever(index_path="algo_repo/rag_index.index", pkl_path="algo_repo/rag_index.pkl"):
    return VectorRetriever(index_path, pkl_path)

class VectorRetriever:
    """
    专门负责向量检索的类
    """
    def __init__(
        self, 
        index_path: str = "algo_repo/rag_index.index", 
        pkl_path: str = "algo_repo/rag_index.pkl", 
        model_name: str = 'BAAI/bge-small-zh-v1.5',
        model_dim: int = 512,
        device: str = "cpu"
    ):
        """
        初始化检索器
        :param index_path: FAISS 索引文件路径
        :param pkl_path: 元数据 pickle 文件路径
        :param model_name: Embedding 模型名称或本地路径
        :param device: 运行设备 ('cpu' 或 'cuda')
        """
        self.index_path = index_path
        self.pkl_path = pkl_path
        
        # 1. 加载 Embedding 模型
        print(f"Loading Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        
        # 加载初始索引
        if os.path.exists(index_path) and os.path.exists(pkl_path):
            self.index = faiss.read_index(index_path)
            with open(pkl_path, 'rb') as f:
                self.docs_list = pickle.load(f)
        else:
            # 如果没有预建索引，初始化一个空的（向量维度需与模型一致，bge-small 是 512）
            self.index = faiss.IndexFlatL2(model_dim)
            self.docs_list = []



    def add_uploaded_files(self, uploaded_files):
        """核心：解析上传的文件流并实时加入向量库"""
        new_texts = []
        for file in uploaded_files:
            ext = os.path.splitext(file.name)[1].lower()
            text = ""
            try:
                if ext == '.pdf':
                    reader = PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                elif ext == '.docx':
                    doc = Document(file)
                    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                elif ext in ('.txt', '.names'):
                    text = file.read().decode("utf-8", errors='ignore')
            except Exception as e:
                st.error(f"读取 {file.name} 出错: {e}")
                continue

            if text.strip():
                # 切片 (Chunking)
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                for i, chunk in enumerate(chunks):
                    new_texts.append({"title": f"用户上传:{file.name}_p{i}", "content": chunk})

        if new_texts:
            # 向量化并加入 FAISS
            embeddings = self.model.encode([d['content'] for d in new_texts], convert_to_numpy=True).astype('float32')
            self.index.add(embeddings)
            self.docs_list.extend(new_texts)
            return len(new_texts)
        return 0

    def search(self, query: str, top_k: int = 3) -> str:
        """
        实现你 call 方法中需要的 .search 接口
        """
        if not self.docs_list or not query or not query.strip():
            return ""

        # 1. 向量化查询词
        # convert_to_numpy=True 配合 .astype('float32') 保证 FAISS 兼容性
        query_vec = self.model.encode([query], convert_to_numpy=True).astype('float32')

        # 2. 检索
        distances, indices = self.index.search(query_vec, top_k)

        # 3. 解析结果并构建 context 字符串
        context_parts = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.docs_list):
                doc_data = self.docs_list[idx]
                title = doc_data.get('title', f"Doc_{idx}")
                content = doc_data.get('content', '')
                
                # 格式化输出，方便 LLM 理解
                part = f"--- 来源: {title} ---\n{content}"
                context_parts.append(part)

        return "\n\n".join(context_parts)

# 这样写可以方便你单独测试 RAG 逻辑是否正常
if __name__ == "__main__":
    # 假设你已经有了索引文件
    try:
        retriever = VectorRetriever()
        test_res = retriever.search("测试问题")
        print("测试搜索结果：\n", test_res)
    except Exception as e:
        print(f"初始化失败，请检查索引文件是否存在: {e}")