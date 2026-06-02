"""短期记忆 — 滑动窗口对话历史管理 + 自动摘要压缩"""

from collections import deque
from typing import Any


class ShortTermMemory:
    """短期记忆：维护对话历史的滑动窗口

    超过窗口大小后，自动将最早的消息压缩为摘要。
    """

    def __init__(self, max_messages: int = 20, summary_max: int = 5):
        self.max_messages = max_messages
        self.summary_max = summary_max  # 超过此数量后触发摘要
        self.messages: list[dict] = []
        self.summary: str = ""

    def add_message(self, role: str, content: str):
        """添加一条消息到短期记忆"""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.summary_max:
            self._maybe_compress()

    def add_user(self, content: str):
        self.add_message("user", content)

    def add_assistant(self, content: str):
        self.add_message("assistant", content)

    def add_tool_result(self, tool_name: str, result: str):
        self.messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "content": result[:500],  # 截断
        })

    def get_history(self) -> list[dict]:
        """获取用于 LLM 的历史消息列表"""
        return list(self.messages)[-self.max_messages:]

    def _maybe_compress(self):
        """当消息过多时，压缩最早的消息为摘要"""
        if len(self.messages) <= self.max_messages:
            return
        excess = self.messages[:self.summary_max]
        summary_text = " | ".join(
            f"[{m['role']}]: {m['content'][:100]}"
            for m in excess if m['role'] != 'tool'
        )
        self.summary = f"[历史摘要] {summary_text[:500]}"
        self.messages = self.messages[self.summary_max:]

    def get_context(self) -> dict:
        """获取完整的短期记忆上下文"""
        return {
            "summary": self.summary,
            "recent_messages": self.get_history(),
            "total_count": len(self.messages),
        }

    def clear(self):
        self.messages.clear()
        self.summary = ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShortTermMemory":
        mem = cls()
        mem.summary = data.get("summary", "")
        mem.messages = data.get("messages", [])
        return mem
