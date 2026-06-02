"""向量记忆（轻量版）— 基于文件的持久化记忆存储

使用 JSON 文件持久化存储历史分析结论和用户偏好。
支持按行业、城市、关键词检索，无需外部向量数据库。
"""

import json
import os
from datetime import datetime
from typing import Any


class VectorMemory:
    """基于文件的轻量级记忆存储

    每条记忆包含：
    - content: 记忆内容
    - tags: 标签列表（行业、城市、岗位等）
    - timestamp: 创建时间
    - weight: 重要性权重
    """

    def __init__(self, storage_path: str = ""):
        if not storage_path:
            storage_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "agent_memory.json"
            )
        self.storage_path = os.path.abspath(storage_path)
        self._memories: list[dict] = []
        self._load()

    def remember(self, content: str, tags: list[str] | None = None,
                 weight: float = 1.0) -> dict:
        """存储一条记忆"""
        memory = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "content": content,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "weight": weight,
        }
        self._memories.append(memory)
        self._save()
        return memory

    def search(self, query: str = "", tags: list[str] | None = None,
               limit: int = 5, min_weight: float = 0.0) -> list[dict]:
        """按关键词和标签检索记忆

        使用简单的关键词匹配 + 标签过滤，返回按相关度排序的结果。
        """
        query_lower = query.lower() if query else ""
        tag_set = set(tags or [])
        scored = []

        for m in self._memories:
            if m["weight"] < min_weight:
                continue

            score = 0.0

            # 标签匹配
            if tag_set:
                overlap = len(tag_set & set(m.get("tags", [])))
                if overlap == 0 and not query_lower:
                    continue
                score += overlap * 3

            # 关键词匹配
            if query_lower:
                content_lower = m.get("content", "").lower()
                if query_lower in content_lower:
                    score += 2
                else:
                    query_words = query_lower.split()
                    match_count = sum(
                        1 for w in query_words if w in content_lower
                    )
                    score += match_count * 1

            if score > 0:
                scored.append((score * m["weight"], m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_recent(self, limit: int = 10) -> list[dict]:
        """获取最近 N 条记忆"""
        sorted_mem = sorted(
            self._memories,
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )
        return sorted_mem[:limit]

    def get_by_tags(self, tags: list[str], limit: int = 10) -> list[dict]:
        """按标签获取记忆"""
        tag_set = set(tags)
        matched = []
        for m in self._memories:
            if tag_set & set(m.get("tags", [])):
                matched.append(m)
        return matched[:limit]

    def forget(self, memory_id: str) -> bool:
        """删除一条记忆"""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.get("id") != memory_id]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def clear(self):
        """清空所有记忆"""
        self._memories.clear()
        self._save()

    def count(self) -> int:
        return len(self._memories)

    def _load(self):
        """从文件加载记忆"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._memories = data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            self._memories = []

    def _save(self):
        """保存记忆到文件"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, ensure_ascii=False, indent=2)
        except IOError:
            pass  # 静默失败，不阻塞主流程

    def to_dict(self) -> dict:
        return {
            "memories": self._memories,
            "count": len(self._memories),
            "storage_path": self.storage_path,
        }
