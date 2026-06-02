"""工作记忆 — 当前分析任务的上下文状态"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    """工作记忆：记录当前分析任务的上下文状态

    包括用户的筛选条件、关注的维度、已经完成的分析步骤等。
    """

    # 基本查询条件
    current_industry: str = "互联网"
    current_city: str = "北京"

    # 用户关注的维度
    focus_positions: list[str] = field(default_factory=list)
    compare_cities: list[str] = field(default_factory=list)
    compare_industries: list[str] = field(default_factory=list)

    # 任务追踪
    current_task: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)

    # 已获取的数据快照
    last_query_result: dict = field(default_factory=dict)

    def update_query(self, industry: str | None = None, city: str | None = None):
        """更新查询条件"""
        if industry:
            self.current_industry = industry
        if city:
            self.current_city = city

    def add_focus_position(self, position: str):
        if position and position not in self.focus_positions:
            self.focus_positions.append(position)

    def set_task(self, task: str):
        self.current_task = task
        self.completed_steps.clear()
        self.pending_steps.clear()

    def complete_step(self, step: str):
        self.completed_steps.append(step)
        if step in self.pending_steps:
            self.pending_steps.remove(step)

    def add_pending_step(self, step: str):
        if step not in self.pending_steps:
            self.pending_steps.append(step)

    def get_summary(self) -> str:
        """生成工作记忆的文本摘要（用于 LLM context）"""
        parts = [
            f"当前查询：{self.current_industry}行业 / {self.current_city}",
        ]
        if self.focus_positions:
            parts.append(f"关注岗位：{', '.join(self.focus_positions)}")
        if self.compare_cities:
            parts.append(f"对比城市：{', '.join(self.compare_cities)}")
        if self.compare_industries:
            parts.append(f"对比行业：{', '.join(self.compare_industries)}")
        if self.current_task:
            parts.append(f"当前任务：{self.current_task}")
        if self.completed_steps:
            parts.append(f"已完成：{' → '.join(self.completed_steps)}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "current_industry": self.current_industry,
            "current_city": self.current_city,
            "focus_positions": self.focus_positions,
            "compare_cities": self.compare_cities,
            "compare_industries": self.compare_industries,
            "current_task": self.current_task,
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingMemory":
        return cls(
            current_industry=data.get("current_industry", "互联网"),
            current_city=data.get("current_city", "北京"),
            focus_positions=data.get("focus_positions", []),
            compare_cities=data.get("compare_cities", []),
            compare_industries=data.get("compare_industries", []),
            current_task=data.get("current_task", ""),
            completed_steps=data.get("completed_steps", []),
            pending_steps=data.get("pending_steps", []),
        )
