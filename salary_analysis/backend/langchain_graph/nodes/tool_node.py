"""工具执行节点 — 自定义 ToolNode

替换 LangGraph 内置 ToolNode，添加：
1. L3 会话级缓存 (session_cache)
2. 只读工具并行执行 (asyncio.gather)
3. 非只读工具串行执行
4. 工具错误的结构化处理
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from langchain_graph.state import AgentState

logger = logging.getLogger("langchain_graph.nodes.tool_node")

# 可并行执行的只读工具集 (同原 core.py READ_ONLY_TOOLS)
READ_ONLY_TOOLS = {
    "query_salary_statistics",
    "query_recruitment_list",
    "compare_cities",
    "compare_industries",
}


def _is_read_only(tool_call: dict) -> bool:
    """判断单个工具调用是否为只读查询"""
    return tool_call.get("name", "") in READ_ONLY_TOOLS


def _get_session_cache(state: AgentState) -> dict:
    """从 state 中获取 session_cache (L3)"""
    return state.get("session_cache", {})


def _update_session_cache(state: AgentState, tool_name: str, args: dict, result: Any):
    """更新 L3 会话缓存

    仅对 query_salary_statistics 生效 (确定性查询)。
    """
    if tool_name != "query_salary_statistics":
        return
    if not isinstance(result, dict):
        return
    if "industry" not in result or "city" not in result:
        return

    cache = dict(state.get("session_cache", {}))
    key = (result["industry"], result["city"])
    cache[key] = result
    return cache


def create_tool_node(tools: list[StructuredTool]) -> Callable:
    """创建工具执行节点

    Args:
        tools: LangChain StructuredTool 列表

    Returns:
        tool_node: 符合 LangGraph 节点签名的 async 函数
            (state: AgentState) -> dict
    """
    # 工具名 → StructuredTool 映射
    tools_map: dict[str, StructuredTool] = {t.name: t for t in tools}

    async def _execute_one(
        tool_call: dict,
        session_cache: dict,
    ) -> tuple:
        """执行单个工具调用

        Args:
            tool_call: 格式为 {"name": str, "args": dict, "id": str, "type": "tool_call"}
            session_cache: L3 会话缓存字典 (可能被修改)

        Returns:
            ToolMessage 实例
        """
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")
        tool = tools_map.get(tool_name)

        if tool is None:
            logger.warning(f"未知工具: {tool_name}")
            return ToolMessage(
                content=json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        # L3 缓存检查 (仅 query_salary_statistics)
        if tool_name == "query_salary_statistics":
            cache_key = (
                tool_args.get("industry", "互联网"),
                tool_args.get("city", "北京"),
            )
            cached = session_cache.get(cache_key)
            if cached is not None:
                return ToolMessage(
                    content=json.dumps(cached, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )

        try:
            # 执行工具
            result = await tool.ainvoke(tool_args)

            # L3 缓存写入
            if tool_name == "query_salary_statistics":
                if isinstance(result, dict) and "industry" in result and "city" in result:
                    cache_key = (result["industry"], result["city"])
                    session_cache[cache_key] = result
                elif isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and "industry" in parsed and "city" in parsed:
                            cache_key = (parsed["industry"], parsed["city"])
                            session_cache[cache_key] = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

            # 序列化结果为字符串 (兼容 ToolMessage.content)
            if isinstance(result, dict):
                content = json.dumps(result, ensure_ascii=False, default=str)[:3000]
            else:
                content = str(result)[:3000]

            return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        except Exception as e:
            logger.error(f"工具 [{tool_name}] 执行失败: {type(e).__name__}: {e}")
            return ToolMessage(
                content=json.dumps(
                    {"error": f"工具 [{tool_name}] 执行失败: {str(e)}"},
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call_id,
                name=tool_name,
            )

    async def tool_node(state: AgentState) -> dict:
        """工具执行节点

        从 state.messages 的最后一条 AIMessage 中提取 tool_calls，
        区分只读/非只读，分别并行或串行执行。
        发射 tool_parallel_start / tool_start / tool_end 事件。

        Args:
            state: 当前 AgentState

        Returns:
            dict with "messages" key containing ToolMessage 列表
        """
        from langchain_graph._dispatch import emit_event
        from langchain_graph.streaming import (
            create_tool_parallel_start_event,
            create_tool_start_event,
            create_tool_end_event,
        )

        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        session_cache = dict(state.get("session_cache", {}))
        tool_calls = last_message.tool_calls

        results: list[ToolMessage] = []

        # 分离只读和非只读工具
        read_only_tools = [tc for tc in tool_calls if _is_read_only(tc)]
        other_tools = [tc for tc in tool_calls if not _is_read_only(tc)]

        # ── 只读工具 → 并行执行 ──
        if read_only_tools:
            names = [tc.get("name", "") for tc in read_only_tools]
            await emit_event(create_tool_parallel_start_event(
                count=len(read_only_tools), names=names,
            ))

            parallel_results = await asyncio.gather(
                *[_execute_one(tc, session_cache) for tc in read_only_tools],
                return_exceptions=True,
            )
            for i, (tc, res) in enumerate(zip(read_only_tools, parallel_results)):
                t_name = tc.get("name", "")
                t_args = tc.get("args", {})

                if isinstance(res, Exception):
                    results.append(ToolMessage(
                        content=json.dumps(
                            {"error": f"工具 [{t_name}] 并行执行失败: {str(res)}"},
                            ensure_ascii=False,
                        ),
                        tool_call_id=tc.get("id", ""),
                        name=t_name,
                    ))
                    await emit_event(create_tool_error_event(t_name, str(res)))
                else:
                    results.append(res)
                    await emit_event(create_tool_result_event(t_name, res.content if hasattr(res, 'content') else ""))

        # ── 非只读工具 → 串行执行（每个发射 tool_start / tool_end）──
        for tc in other_tools:
            t_name = tc.get("name", "")
            t_args = tc.get("args", {})

            await emit_event(create_tool_start_event(t_name, t_args))

            t0 = time.time()
            result = await _execute_one(tc, session_cache)
            elapsed = time.time() - t0

            if hasattr(result, "content") and isinstance(result.content, str):
                try:
                    parsed = json.loads(result.content)
                    if isinstance(parsed, dict) and "error" in parsed:
                        await emit_event(create_tool_error_event(t_name, parsed["error"]))
                    else:
                        await emit_event(create_tool_end_event(
                            t_name, elapsed, result.content[:300],
                        ))
                        await emit_event(create_tool_result_event(t_name, result.content))
                except (json.JSONDecodeError, TypeError):
                    await emit_event(create_tool_end_event(t_name, elapsed, result.content[:300]))
                    await emit_event(create_tool_result_event(t_name, result.content))
            else:
                await emit_event(create_tool_end_event(t_name, elapsed))

            results.append(result)

        # 更新 state 中的 session_cache (需要在返回的 dict 中包含)
        update: dict = {"messages": results}
        # 仅在 session_cache 有变化时更新
        current_cache = state.get("session_cache", {})
        if len(session_cache) > len(current_cache):
            update["session_cache"] = session_cache

        return update

    return tool_node
