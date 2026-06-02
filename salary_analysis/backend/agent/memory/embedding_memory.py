"""向量记忆（增强版）— 混合检索 + 去重合并 + 重排序

架构：
┌──────────────────────────────────────────┐
│            EmbeddingMemory                │
│  ┌──────────────┐  ┌───────────────────┐  │
│  │  Semantic    │  │  BM25 (jieba分词)  │  │
│  │  (Dense)     │  │  (Sparse)         │  │
│  └──────┬───────┘  └───────┬───────────┘  │
│         └────────┬─────────┘               │
│              ┌────┴────┐                   │
│              │  Fusion │ α=0.6             │
│              └────┬────┘                   │
│              ┌────┴────┐                   │
│              │ Reranker│(可选 CrossEncoder)│
│              └─────────┘                   │
│  ┌──────────────────────────────────────┐  │
│  │  remember() with dedup (阈值 >0.85)  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
"""

import logging
import os
import re

import numpy as np
import jieba

from .vector import VectorMemory

logger = logging.getLogger("agent.memory.embedding")


class BM25Index:
    """BM25 稀疏检索索引

    使用 jieba 分词 + rank_bm25，在记忆写入时重建索引。
    没有持久化——重建一次 O(n log n) 对 <1000 条记忆可忽略。
    """

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self._dirty = True

    def build(self, memories: list[dict]):
        """从记忆列表重建 BM25 索引"""
        self.corpus = [m.get("content", "") for m in memories if m.get("content")]
        if not self.corpus:
            self.bm25 = None
            return
        tokenized = [list(jieba.cut(c)) for c in self.corpus]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(tokenized)
        self._dirty = False

    def search(self, query: str, memories: list[dict], top_k: int = 50) -> dict[str, float]:
        """返回 {记忆文本: BM25分数} 映射"""
        if self.bm25 is None or not query:
            return {}

        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25.get_scores(tokenized_query)

        result = {}
        for i, score in enumerate(doc_scores):
            if score > 0 and i < len(memories):
                c = memories[i].get("content", "")
                result[c] = float(score)
        return result


class Reranker:
    """交叉编码器重排序（可选）

    首次使用时延迟加载 CrossEncoder 模型。
    模型不存在或加载失败则跳过重排序。
    """

    def __init__(self, model_name: str = ""):
        self._model_name = model_name or "cross-encoder/stsb-distilroberta-base"
        self._model = None
        self._loaded = False
        self._tried = False

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """对候选列表重排序，返回重排后的 Top-K"""
        model = self._get_model()
        if model is None or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, m.get("content", "")) for m in candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            for m, s in zip(candidates, scores):
                m["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as e:
            logger.warning("重排序失败，使用原始排序: %s", e)

        return candidates[:top_k]

    def _get_model(self):
        if self._tried:
            return self._model if self._loaded else None
        self._tried = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            self._loaded = True
            logger.info("重排序模型 %s 加载完成", self._model_name)
        except Exception as e:
            logger.warning("重排序模型加载失败，跳过重排序: %s", e)
        return self._model if self._loaded else None


class EmbeddingMemory(VectorMemory):
    """增强版语义向量记忆

    在 VectorMemory 基础上增加：
    1. Dense + BM25 混合检索（权重 α=0.6）
    2. remember() 去重合并（相似度 >0.85 时合并而非追加）
    3. 可选 CrossEncoder 重排序（top-20 → top-3）
    4. 模型加载失败时静默回退到父类关键词搜索
    """

    def __init__(self, storage_path: str = "", model_name: str = ""):
        super().__init__(storage_path)
        self._model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._model = None
        self._model_loaded = False
        self._model_tried = False

        # 混合检索组件
        self._bm25 = BM25Index()
        self._dense_weight = 0.6        # Dense 权重 α
        self._sparse_weight = 0.4       # Sparse 权重 1-α

        # 去重配置
        self._dedup_threshold = 0.85    # 相似度超过此值则合并
        self._dedup_counter = 0         # 合并统计

        # 重排序（可选）
        self._reranker = Reranker()

    @property
    def dedup_stats(self) -> dict:
        """去重统计"""
        return {"merged_count": self._dedup_counter, "threshold": self._dedup_threshold}

    # ── 公开接口 ──

    def remember(self, content: str, tags: list[str] | None = None,
                 weight: float = 1.0) -> dict:
        """存储一条记忆（带去重合并）

        流程：
        1. 计算 content 的 embedding
        2. 检查已有记忆的余弦相似度
        3. 相似度 > threshold → 合并到已有记忆（更新权重+时间戳）
        4. 否则创建新记忆
        """
        embedding = self._compute_embedding(content)

        # 尝试去重合并
        if embedding is not None and self._memories:
            merged = self._try_dedup(content, embedding, tags, weight)
            if merged:
                return merged

        # 创建新记忆
        memory = super().remember(content, tags, weight)
        if embedding is not None:
            memory["embedding"] = embedding.tolist()
            for m in self._memories:
                if m.get("id") == memory["id"]:
                    m["embedding"] = memory["embedding"]
                    break
            self._save()

        # 标记 BM25 索引需要重建
        self._bm25._dirty = True
        return memory

    def search(self, query: str = "", tags: list[str] | None = None,
               limit: int = 5, min_weight: float = 0.0) -> list[dict]:
        """混合检索：Dense + BM25 融合 → 可选重排序

        评分公式：
        score = α × normalize(dense_score) + (1-α) × normalize(sparse_score)
        score = score × 0.7 + tag_match × 0.3
        score = score × weight
        """
        if not query:
            return super().search(query, tags, limit, min_weight)

        query_vec = self._compute_embedding(query)
        if query_vec is None:
            return super().search(query, tags, limit, min_weight)

        # 1. Dense 语义检索
        tag_set = set(tags or [])
        dense_scores = {}  # content -> score
        sparse_scores = {}  # content -> score

        for m in self._memories:
            if m["weight"] < min_weight:
                continue
            m_emb = m.get("embedding")
            if m_emb is not None:
                score = self._cosine_similarity(query_vec, np.array(m_emb, dtype=np.float32))
                if score > 0:
                    dense_scores[m["content"]] = score

        # 2. BM25 稀疏检索
        if self._bm25._dirty:
            self._bm25.build(self._memories)
        sparse_scores = self._bm25.search(query, self._memories)

        # 3. 融合打分
        dense_max = max(dense_scores.values()) if dense_scores else 1.0
        sparse_max = max(sparse_scores.values()) if sparse_scores else 1.0

        scored = []
        for m in self._memories:
            if m["weight"] < min_weight:
                continue

            c = m["content"]
            d_score = dense_scores.get(c, 0) / dense_max if dense_max > 0 else 0
            s_score = sparse_scores.get(c, 0) / sparse_max if sparse_max > 0 else 0

            # 融合：Dense + Sparse
            hybrid = self._dense_weight * d_score + self._sparse_weight * s_score

            # Tag 匹配
            tag_score = 0.0
            if tag_set:
                m_tags = set(m.get("tags", []))
                if tag_set & m_tags:
                    tag_score = len(tag_set & m_tags) / max(len(tag_set), 1)

            combined = (hybrid * 0.7 + tag_score * 0.3) * m["weight"]
            if combined > 0:
                m["_score"] = round(combined, 4)
                m["_dense"] = round(d_score, 4)
                m["_sparse"] = round(s_score, 4)
                scored.append(m)

        scored.sort(key=lambda x: x["_score"], reverse=True)
        top_n = scored[:min(limit * 4, 20)]  # 宽召回给 rerank

        # 4. 重排序
        if len(top_n) >= 3:
            top_n = self._reranker.rerank(query, top_n, top_k=limit)

        return top_n[:limit]

    # ── 去重 ──

    def _try_dedup(self, content: str, embedding: np.ndarray,
                   tags: list[str] | None, weight: float) -> dict | None:
        """尝试去重合并：
        如果新内容与已有记忆语义相似度 > threshold，则合并到最相似的记忆
        """
        best_sim = 0.0
        best_mem = None
        vec = np.array(embedding, dtype=np.float32)

        for m in self._memories:
            m_emb = m.get("embedding")
            if m_emb is None:
                continue
            sim = self._cosine_similarity(vec, np.array(m_emb, dtype=np.float32))
            if sim > best_sim:
                best_sim = sim
                best_mem = m

        if best_sim > self._dedup_threshold and best_mem:
            # 合并：更新权重 + 时间戳 + tags
            best_mem["weight"] = max(best_mem["weight"], weight)
            from datetime import datetime
            best_mem["timestamp"] = datetime.now().isoformat()
            best_mem["access_count"] = best_mem.get("access_count", 0) + 1

            # 合并 tags
            if tags:
                existing_tags = set(best_mem.get("tags", []))
                existing_tags.update(tags)
                best_mem["tags"] = list(existing_tags)

            self._dedup_counter += 1
            self._save()

            logger.info("去重合并: sim=%.3f, old='%s...', new='%s...'",
                        best_sim, best_mem["content"][:30], content[:30])
            return best_mem

        return None

    # ── 内部方法 ──

    def _compute_embedding(self, text: str) -> np.ndarray | None:
        model = self._get_model()
        if model is None:
            return None
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            logger.warning("Embedding 计算失败: %s", e)
            return None

    def _get_model(self):
        if self._model_tried:
            return self._model if self._model_loaded else None
        self._model_tried = True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("正在加载 embedding 模型: %s ...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._model_loaded = True
            logger.info("Embedding 模型加载完成 (dim=%d)", self._model.get_embedding_dimension())
        except Exception as e:
            logger.warning("Embedding 模型加载失败，回退到关键词搜索: %s", e)
            self._model = None
            self._model_loaded = False
        return self._model if self._model_loaded else None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
