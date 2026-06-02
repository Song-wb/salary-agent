"""Guardrails 节点 — 输入安全检查和输出质量检查

作为 LangGraph 的条件节点，在 Agent 处理前后进行护栏检查：

    用户输入 → [InputGuard] → 通过 → Agent 处理 → [OutputGuard] → 通过 → 输出
                             失败 ↗              失败 ↗

InputGuard: 检测 prompt injection 和敏感信息泄露
OutputGuard: 检查回答是否包含空输出或主观表述

用法:
    from langchain_graph.nodes.guardrails_node import create_guardrails_node
    guard_node, guard_condition = create_guardrails_node()

    # 在图中插入 guard 节点
    builder.add_node("input_guard", guard_node["input"])
    builder.add_node("output_guard", guard_node["output"])
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from langchain_graph.state import AgentState
from langchain_graph._dispatch import emit_event

logger = logging.getLogger("langchain_graph.nodes.guardrails")


def create_guardrails_node() -> dict[str, Callable]:
    """创建 Guardrails 节点对 (input + output)

    Returns:
        dict with:
            "input": async (state) -> state_update  输入检查节点
            "output": async (state) -> state_update 输出检查节点

    输入节点检查 state.user_message 是否包含注入或敏感信息。
    如果检测到问题，设置 state.guardrails_passed = False
    并注入 corrective 消息。

    输出节点检查最后一条 AIMessage 的内容质量。
    如果检测到问题，设置 state.guardrails_output_issues。
    """

    async def input_guard_node(state: AgentState) -> dict:
        """输入护栏节点 — 检查用户消息"""
        from agent.guardrails import InputGuard

        user_message = state.get("user_message", "")
        if not user_message:
            return {}

        guard_result = InputGuard.check(user_message)

        if guard_result["passed"]:
            await emit_event({
                "type": "thought",
                "content": "✅ 输入安全检查通过",
            })
            return {"guardrails_passed": True}

        # 检测到问题
        issues = guard_result["issues"]
        await emit_event({
            "type": "reflection",
            "issues": issues,
            "reflection_depth": 1,
            "max_depth": 1,
            "stage": "input_guard",
        })

        logger.warning(f"输入安全检查未通过: {issues}")
        return {
            "guardrails_passed": False,
            "guardrails_issues": issues,
            "messages": [
                HumanMessage(
                    content=(
                        "⚠️ 系统检测到您的输入可能包含不安全内容。"
                        "请重新描述您的问题，不要尝试修改或绕过系统指令。"
                    )
                ),
            ],
        }

    async def output_guard_node(state: AgentState) -> dict:
        """输出护栏节点 — 检查最终回答质量"""
        from agent.guardrails import OutputGuard

        messages = state.get("messages", [])
        if not messages:
            return {"guardrails_output_issues": []}

        last_msg = messages[-1]
        content = ""
        if hasattr(last_msg, "content") and last_msg.content:
            content = last_msg.content

        if not content:
            return {"guardrails_output_issues": []}

        # 检查是否有数据可用 (state 中是否有 ToolMessage)
        has_data = any(
            hasattr(m, "content") and m.content
            for m in messages
            if hasattr(m, "type") and m.type == "tool" or hasattr(m, "name")
        )

        guard_result = OutputGuard.check(content, data_available=has_data)

        issues = guard_result["issues"]
        if not guard_result["passed"]:
            await emit_event({
                "type": "reflection",
                "issues": issues,
                "reflection_depth": 1,
                "max_depth": 1,
                "stage": "output_guard",
            })

        return {"guardrails_output_issues": issues}

    return {
        "input": input_guard_node,
        "output": output_guard_node,
    }


def input_guard_condition(state: AgentState) -> str:
    """条件边路由: 输入检查是否通过

    Returns:
        "process" — 安全，继续 Agent 处理
        "rejected" — 不安全，直接返回拒绝消息
    """
    if state.get("guardrails_passed", True) is False:
        return "rejected"
    return "process"


def output_guard_condition(state: AgentState) -> str:
    """条件边路由: 输出检查是否通过

    Returns:
        "end" — 质量合格
        "revise" — 有问题，需要修正
    """
    issues = state.get("guardrails_output_issues", [])
    serious = [i for i in issues if i.get("severity") in ("high", "medium")]
    if serious:
        return "revise"
    return "end"
