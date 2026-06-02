"""Agent 节点工厂 — LLM 调用（流式增强版）

使用 llm.astream() 替代 llm.ainvoke()，确保流式文本事件
通过 on_chat_model_stream 正确发送给前端 SSE。

当 LLM 输出推理内容（有 tool_calls）时，
额外发射 thought 自定义事件供前端显示。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessageChunk

from langchain_graph.state import AgentState


def _build_full_messages(state: AgentState) -> list:
    """组装完整的 LLM 消息列表

    顺序: SystemPrompt(角色定义) + 短期记忆摘要 + 工作记忆上下文 + 历史消息
    """
    system_prompt = state.get("system_prompt", "")
    short_term = state.get("short_term_summary", "")
    industry = state.get("current_industry", "互联网")
    city = state.get("current_city", "北京")

    # 构建增强的 system prompt
    enhanced_system = system_prompt
    context_parts = []

    if short_term:
        context_parts.append(f"历史对话摘要：{short_term}")

    context_parts.append(f"当前查询：{industry}行业 / {city}")

    focus = state.get("focus_positions", [])
    if focus:
        context_parts.append(f"关注岗位：{', '.join(focus)}")

    if context_parts:
        enhanced_system += "\n\n当前上下文：\n" + "\n".join(context_parts)

    messages = [SystemMessage(content=enhanced_system)]

    # 添加已有的消息 (用户提问、工具结果等)
    for msg in state.get("messages", []):
        messages.append(msg)

    return messages


def create_agent_node(llm: BaseChatModel) -> Callable:
    """创建 LLM 调用节点（流式）

    使用 astream() 确保 streaming callback 被正确触发，
    使 LangGraph 的 astream_events 能捕获 on_chat_model_stream 事件。

    Args:
        llm: 已配置的 LangChain ChatModel 实例 (带熔断保护)

    Returns:
        agent_node: 符合 LangGraph 节点签名的 async 函数
            (state: AgentState) -> dict
    """

    async def agent_node(state: AgentState) -> dict:
        """LLM 调用节点（流式）

        流式收集 LLM 输出，同时发射 SSE 文本事件。
        最后返回完整的 AIMessage（含 tool_calls）。
        """
        from langchain_graph._dispatch import emit_event

        full_messages = _build_full_messages(state)

        # 使用流式调用，确保 on_chat_model_stream 事件被触发
        full_response = None
        has_tool_calls = False

        async for chunk in llm.astream(full_messages):
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk  # type: ignore

            # 检测是否有 tool_call
            if chunk.tool_call_chunks:
                has_tool_calls = True

        if full_response is None:
            return {"messages": []}

        # 转换为 AIMessage（AIMessageChunk → AIMessage）
        final_message = full_response.to_message() if hasattr(full_response, 'to_message') else full_response

        # 如果 LLM 输出了推理内容同时要调工具，发射 thought 事件
        if hasattr(final_message, "content") and final_message.content and has_tool_calls:
            await emit_event({
                "type": "thought",
                "content": f"🤔 {final_message.content[:500]}",
            })

        return {"messages": [final_message]}

    return agent_node
