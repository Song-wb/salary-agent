"""EmbeddingMemory → LangChain VectorStore 适配器

将项目现有的 EmbeddingMemory（混合检索 + 去重 + 重排序）
包装为 LangChain VectorStore 接口，使其可以作为
LangGraph 的 retriever 工具使用。

用法:
    from langchain_graph.memory.vector_store import MemoryVectorStore
    from agent.memory import EmbeddingMemory

    memory = EmbeddingMemory()
    store = MemoryVectorStore(memory)

    # 作为 retriever 使用
    retriever = store.as_retriever(search_kwargs={"k": 5})
    docs = await retriever.ainvoke("北京互联网薪资")

    # 作为工具集成到 agent 中
    from langchain.tools import create_retriever_tool
    tool = create_retriever_tool(retriever, "search_memory", "搜索历史记忆")
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


class MemoryVectorStore(VectorStore):
    """将 EmbeddingMemory 包装为 LangChain VectorStore

    适配器模式：不替换原有 EmbeddingMemory 实现，
    而是在其上层提供 LangChain 期望的接口。

    关键映射:
        EmbeddingMemory.remember(content, tags) → VectorStore.add_texts(...)
        EmbeddingMemory.search(query, tags, limit) → VectorStore.similarity_search(...)
    """

    def __init__(
        self,
        memory_instance,
        embedding: Optional[Embeddings] = None,
    ):
        """初始化适配器

        Args:
            memory_instance: 现有的 EmbeddingMemory 或 VectorMemory 实例
            embedding: 可选的 LangChain Embeddings 实例 (用于接口兼容)
                      实际检索使用 memory 内部的 embedding 模型
        """
        self._memory = memory_instance
        self._embedding = embedding

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    # ── 写入接口 ──

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """添加文本到记忆存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表 (支持 tags, weight 字段)
            **kwargs: 透传给 EmbeddingMemory.remember()

        Returns:
            记忆 ID 列表
        """
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            tags = meta.get("tags", kwargs.get("tags", []))
            weight = meta.get("weight", kwargs.get("weight", 1.0))

            result = self._memory.remember(
                content=text,
                tags=tags if isinstance(tags, list) else [str(tags)],
                weight=float(weight),
            )
            ids.append(str(result.get("id", "")))
        return ids

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """异步添加文本 (直接同步执行，因为 EmbeddingMemory 是同步的)"""
        return self.add_texts(texts, metadatas, **kwargs)

    # ── 检索接口 ──

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """相似度搜索，返回 Document 列表

        Args:
            query: 搜索查询
            k: 返回结果数量
            **kwargs: 支持 tags (list[str]), min_weight (float)

        Returns:
            Document 列表 (metadata 包含 source, score, tags 等)
        """
        tags = kwargs.get("tags")
        min_weight = kwargs.get("min_weight", 0.0)

        results = self._memory.search(
            query=query,
            tags=tags if isinstance(tags, list) else None,
            limit=k,
            min_weight=min_weight,
        )

        return [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "id": r.get("id", ""),
                    "score": r.get("_score", 0),
                    "tags": r.get("tags", []),
                    "weight": r.get("weight", 1.0),
                    "timestamp": r.get("timestamp", ""),
                    "source": "embedding_memory",
                },
            )
            for r in results
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """异步相似度搜索"""
        return self.similarity_search(query, k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """按向量搜索 (使用 EmbeddingMemory 内部的向量检索)"""
        # EmbeddingMemory 不直接暴露按向量搜索的接口，
        # 这里使用一个技巧：用空查询 + 标签过滤做降级
        return self.similarity_search("", k, **kwargs)

    # ── 删除接口 ──

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """删除记忆 (EmbeddingMemory 不支持单条删除，清空全部)"""
        if ids is not None:
            # EmbeddingMemory 不提供单条删除 API，记录 warning
            import logging
            logger = logging.getLogger("langchain_graph.memory.vector_store")
            logger.warning("MemoryVectorStore 不支持单条删除，忽略 delete(ids=%s)", ids)
        else:
            self._memory.clear()

    # ── 其他接口 ──

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MemoryVectorStore":
        """从文本列表创建实例

        通过 kwargs 传递 memory_instance 参数。
        """
        memory_instance = kwargs.get("memory_instance")
        if memory_instance is None:
            from agent.memory import EmbeddingMemory
            memory_instance = EmbeddingMemory()

        store = cls(memory_instance, embedding)
        if texts:
            store.add_texts(texts, metadatas)
        return store

    def get_memory_instance(self):
        """获取原始 EmbeddingMemory 实例 (用于直接访问原有 API)"""
        return self._memory
