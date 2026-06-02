"""AgentState — 所有 LangGraph 图共享的状态模式"""

from __future__ import annotations

import time
from typing import Annotated, Any, Optional

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """所有 Agent 图共享的状态

    LangGraph 的 StateGraph 依赖 TypedDict 来定义状态结构。
    messages 字段使用 add_messages 归约器(reducer)，
    确保新消息自动追加到列表中而非覆盖。
    """

    # ── 核心消息列表 (LangGraph 自动管理) ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 任务信息 ──
    task_type: str                    # simple / analysis / report / chart / complex
    user_message: str                 # 原始用户输入
    system_prompt: str                # 当前 Agent 的角色系统提示词

    # ── 工作记忆 (原 WorkingMemory 字段) ──
    current_industry: str             # 当前行业，默认 "互联网"
    current_city: str                 # 当前城市，默认 "北京"
    focus_positions: list[str]        # 关注的岗位列表
    compare_cities: list[str]         # 对比城市列表
    compare_industries: list[str]     # 对比行业列表

    # ── 短期记忆上下文 (原 ShortTermMemory) ──
    short_term_summary: str           # 历史对话摘要

    # ── 反射状态 (原 Reflector) ──
    reflection_count: int             # 当前循环已反射次数
    max_reflection: int               # 最大反射次数 (默认 2)
    contradiction_cache: dict         # 跨工具数据矛盾检测缓存

    # ── 会话级缓存 (L3, 同原 session_cache) ──
    session_cache: dict               # key: (industry, city) → statistics dict

    # ── 可观测性 (原 TraceContext) ──
    trace_id: str                     # 请求追踪 ID
    start_time: float                 # 请求开始时间戳


def make_initial_state(
    user_message: str = "",
    system_prompt: str = "",
    task_type: str = "simple",
    industry: str = "互联网",
    city: str = "北京",
    short_term_summary: str = "",
    trace_id: str = "",
) -> dict:
    """创建 AgentState 的初始值

    使用工厂函数而非直接构造 TypedDict 以确保所有字段都有缺省值。
    """
    return {
        "messages": [],
        "task_type": task_type,
        "user_message": user_message,
        "system_prompt": system_prompt,
        "current_industry": industry,
        "current_city": city,
        "focus_positions": [],
        "compare_cities": [],
        "compare_industries": [],
        "short_term_summary": short_term_summary,
        "reflection_count": 0,
        "max_reflection": 2,
        "contradiction_cache": {},
        "session_cache": {},
        "trace_id": trace_id or f"lg_{int(time.time())}",
        "start_time": time.time(),
    }
