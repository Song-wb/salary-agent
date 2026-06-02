"""反射节点 — 自纠错检查

包装现有的 Reflector 类，在 ToolNode 执行后检查工具结果，
或在 LLM 给出最终回答后检查回答质量。

作为 LangGraph 节点使用，通过条件边决定下一步：
- 反射通过 → END
- 反射失败且未达上限 → 继续 Agent 循环
- 反射失败且已达上限 → END (避免无限循环)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import ToolMessage, HumanMessage

from langchain_graph.state import AgentState

logger = logging.getLogger("langchain_graph.nodes.reflection_node")


def create_reflection_node() -> Callable:
    """创建反射检查节点

    使用现有的 agent.reflection.reflector.Reflector 进行确定性检查。
    检查分为两类：
    1. 工具结果检查 (有 ToolMessage 的最近步骤) — 错误/空结果/矛盾
    2. 最终回答检查 (LLM 直接回答) — 空回答/主观表述/缺少数据

    Returns:
        reflection_node: 符合 LangGraph 节点签名的 async 函数
            (state: AgentState) -> dict
    """

    async def reflection_node(state: AgentState) -> dict:
        """反射检查节点

        根据当前 state 判断执行哪种检查：
        - 如果最后消息是 ToolMessage → 检查工具结果
        - 如果最后消息是 AIMessage → 检查最终回答质量
        检测到问题时发射 reflection 事件。

        Returns:
            dict with:
            - messages (可选): corrective messages (反射失败时)
            - reflection_count: 更新后的反射次数
            - needs_reflection: 是否需要继续循环
        """
        from langchain_graph._dispatch import emit_event
        from agent.reflection.reflector import Reflector

        reflector = Reflector()
        messages = state.get("messages", [])
        reflection_count = state.get("reflection_count", 0)
        max_reflection = state.get("max_reflection", 2)

        if reflection_count >= max_reflection:
            return {"reflection_count": reflection_count, "needs_reflection": False}

        if not messages:
            return {"reflection_count": reflection_count, "needs_reflection": False}

        last_msg = messages[-1]
        issues_data = []

        # ── 情况 1: 最后消息是 ToolMessage → 检查工具结果 ──
        if isinstance(last_msg, ToolMessage):
            # 从 state 中提取工具步骤信息 (最后 N 个 ToolMessage)
            tool_steps = _extract_tool_steps(messages)

            if not tool_steps:
                return {"reflection_count": reflection_count, "needs_reflection": False}

            ref_result = reflector.check_tool_results(tool_steps)

            if ref_result.passed:
                return {"reflection_count": reflection_count, "needs_reflection": False}

            # 构造 issues_data 用于事件
            issues_data = [
                {"type": i.check_type, "severity": i.severity,
                 "message": i.message, "details": i.details}
                for i in ref_result.issues
            ]

            # 发射 reflection 事件
            await emit_event({
                "type": "reflection",
                "issues": issues_data,
                "reflection_depth": reflection_count + 1,
                "max_depth": max_reflection,
                "stage": "tool_result",
            })

            # 反射失败 → 注入 corrective 消息
            logger.info(
                f"反射检测到 {len(ref_result.issues)} 个工具结果问题, "
                f"深度 {reflection_count + 1}/{max_reflection}"
            )
            update = {
                "reflection_count": reflection_count + 1,
                "needs_reflection": True,
                "messages": ref_result.corrective_messages,
            }
            return update

        # ── 情况 2: 最后消息是 AIMessage (无 tool_calls) → 检查最终回答 ──
        elif hasattr(last_msg, "content") and last_msg.content:
            content = last_msg.content
            ref_result = reflector.check_final_answer(content)

            if ref_result.passed:
                return {"reflection_count": reflection_count, "needs_reflection": False}

            # 构造 issues_data 用于事件
            issues_data = [
                {"type": i.check_type, "severity": i.severity,
                 "message": i.message, "details": i.details}
                for i in ref_result.issues
            ]

            # 发射 reflection 事件
            await emit_event({
                "type": "reflection",
                "issues": issues_data,
                "reflection_depth": reflection_count + 1,
                "max_depth": max_reflection,
                "stage": "final_answer",
            })

            logger.info(
                f"反射检测到 {len(ref_result.issues)} 个回答质量问题, "
                f"深度 {reflection_count + 1}/{max_reflection}"
            )
            update = {
                "reflection_count": reflection_count + 1,
                "needs_reflection": True,
                "messages": ref_result.corrective_messages,
            }
            return update

        # 未知情况 → 通过
        return {"reflection_count": reflection_count, "needs_reflection": False}

    # 同步路由函数 (LangGraph 条件边不能是 async)
    def should_continue(state: AgentState) -> str:
        """条件边路由: 反射通过 → end, 失败 → agent"""
        if state.get("needs_reflection", False):
            return "agent"
        return "end"

    # 将路由函数附加到节点 (便于 react_graph.py 引用)
    reflection_node.should_continue = should_continue  # type: ignore

    return reflection_node


def _extract_tool_steps(messages: list) -> list[dict]:
    """从消息列表中提取最近的工具步骤信息 (用于反射检查)

    从最后一条 ToolMessage 向前搜索，收集工具步骤，
    格式兼容 Reflector.check_tool_results() 的输入。

    Returns:
        list of dict-like objects (兼容 step.tool_name, step.tool_result 等访问)
    """
    steps = []
    # 从后往前搜索 ToolMessages
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        steps.append(_ToolStepAdapter(msg))
        if len(steps) >= 10:  # 最多检查最近 10 步
            break

    return steps


class _ToolStepAdapter:
    """将 ToolMessage 适配为 Reflector 期望的 step 接口

    Reflector 需要 step.tool_name, step.tool_result 等属性。
    """

    def __init__(self, msg: ToolMessage):
        self._msg = msg

    @property
    def tool_name(self) -> str:
        return self._msg.name or ""

    @property
    def tool_result(self) -> str:
        return self._msg.content or ""

    @property
    def tool_args(self) -> dict:
        return {}
