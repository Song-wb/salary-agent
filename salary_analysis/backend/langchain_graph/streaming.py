"""LangGraph 事件 → SSE 事件映射 (Phase 4 增强版)

前端消费的 11 种 SSE 事件类型及映射来源：

SSE 事件            │ LangGraph 来源              │ 产生节点
────────────────────┼────────────────────────────┼────────────────────
task_type           │ Supervisor.classify() 输出   │ Supervisor
thought             │ dispatch_custom_event       │ agent_node
tool_parallel_start │ dispatch_custom_event       │ tool_node (只读并行)
tool_call           │ on_tool_start               │ LangGraph 原生
tool_start          │ dispatch_custom_event       │ tool_node (带 tool 名)
tool_end            │ dispatch_custom_event       │ tool_node (带耗时)
tool_error          │ on_tool_end (含 error)      │ LangGraph 原生
tool_result         │ on_tool_end                 │ LangGraph 原生
reflection          │ dispatch_custom_event       │ reflection_node
text                │ on_chat_model_stream        │ LangGraph 原生
done                │ create_done_event() 构造    │ Supervisor

用法:
    from langchain_graph.streaming import langgraph_events_to_sse
    async for sse_event in langgraph_events_to_sse(graph.astream_events(...)):
        yield {"data": json.dumps(sse_event, ensure_ascii=False)}
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator


# ── 事件类型常量 (前端消费的全部 SSE 事件) ─────────────────────────

EVENT_TASK_TYPE = "task_type"
EVENT_THOUGHT = "thought"
EVENT_TOOL_PARALLEL_START = "tool_parallel_start"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_END = "tool_end"
EVENT_TOOL_ERROR = "tool_error"
EVENT_TOOL_RESULT = "tool_result"
EVENT_REFLECTION = "reflection"
EVENT_TEXT = "text"
EVENT_DONE = "done"

ALL_EVENTS = {
    EVENT_TASK_TYPE,
    EVENT_THOUGHT,
    EVENT_TOOL_PARALLEL_START,
    EVENT_TOOL_CALL,
    EVENT_TOOL_START,
    EVENT_TOOL_END,
    EVENT_TOOL_ERROR,
    EVENT_TOOL_RESULT,
    EVENT_REFLECTION,
    EVENT_TEXT,
    EVENT_DONE,
}


# ── 事件创建工厂 ─────────────────────────────────────────────────


def create_task_type_event(task_type: str) -> dict:
    """创建 task_type 事件 — 任务分类结果"""
    return {"type": EVENT_TASK_TYPE, "task_type": task_type}


def create_thought_event(content: str) -> dict:
    """创建 thought 事件 — AI 思考过程"""
    return {"type": EVENT_THOUGHT, "content": content}


def create_tool_parallel_start_event(
    count: int, names: list[str],
) -> dict:
    """创建 tool_parallel_start 事件 — 并行工具开始"""
    return {
        "type": EVENT_TOOL_PARALLEL_START,
        "count": count,
        "names": names,
    }


def create_tool_call_event(name: str, args: dict, thought: str = "") -> dict:
    """创建 tool_call 事件 — 工具调用"""
    return {
        "type": EVENT_TOOL_CALL,
        "name": name,
        "args": args,
        "thought": thought,
    }


def create_tool_start_event(tool: str, args: dict, thought: str = "") -> dict:
    """创建 tool_start 事件 — 工具开始执行 (tool 字段替代 name)"""
    return {
        "type": EVENT_TOOL_START,
        "tool": tool,
        "args": args,
        "thought": thought,
    }


def create_tool_end_event(
    tool: str,
    duration_s: float,
    result_summary: str = "",
) -> dict:
    """创建 tool_end 事件 — 工具执行结束 (含耗时)"""
    return {
        "type": EVENT_TOOL_END,
        "tool": tool,
        "duration_s": round(duration_s, 2),
        "result_summary": result_summary[:300],
    }


def create_tool_error_event(tool: str, error: str) -> dict:
    """创建 tool_error 事件 — 工具执行错误"""
    return {
        "type": EVENT_TOOL_ERROR,
        "tool": tool,
        "error": str(error)[:500],
    }


def create_tool_result_event(name: str, result: str) -> dict:
    """创建 tool_result 事件 — 工具返回结果"""
    return {
        "type": EVENT_TOOL_RESULT,
        "name": name,
        "result": str(result)[:3000],
    }


def create_reflection_event(
    issues: list[dict],
    reflection_depth: int,
    max_depth: int,
) -> dict:
    """创建 reflection 事件 — 反射检测到问题、正在修正"""
    return {
        "type": EVENT_REFLECTION,
        "issues": issues,
        "reflection_depth": reflection_depth,
        "max_depth": max_depth,
    }


def create_text_event(content: str) -> dict:
    """创建 text 事件 — 流式文本块"""
    return {"type": EVENT_TEXT, "content": content}


def create_done_event(
    content: str = "",
    steps: int = 0,
    token_usage: dict | None = None,
    total_time: float = 0.0,
) -> dict:
    """创建 done 事件 — 处理完成"""
    return {
        "type": EVENT_DONE,
        "content": content,
        "steps": steps,
        "token_usage": token_usage or {},
        "total_time": total_time,
    }


# ── 主映射函数 ───────────────────────────────────────────────────


async def langgraph_events_to_sse(
    langgraph_events: AsyncGenerator[dict, None],
) -> AsyncGenerator[dict, None]:
    """将 LangGraph astream_events 转换为前端 SSE 格式

    Args:
        langgraph_events: LangGraph 的事件异步生成器
            (来自 compiled_graph.astream_events(input, version="v2"))

    Yields:
        dict: 兼容前端 SSE 格式的事件字典

    LangGraph 原生事件:
        on_chat_model_stream  → text
        on_tool_start         → tool_call
        on_tool_end           → tool_result / tool_error
        on_custom_event       → 透传 (thought, reflection, tool_start, tool_end 等)

    用法 (FastAPI):
        from sse_starlette.sse import EventSourceResponse

        async def event_generator():
            lg_events = graph.astream_events(state, config, version="v2")
            async for sse_event in langgraph_events_to_sse(lg_events):
                yield {"data": json.dumps(sse_event, ensure_ascii=False)}

        return EventSourceResponse(event_generator())
    """
    async for event in langgraph_events:
        kind = event.get("event", "")
        name = event.get("name", "")
        data = event.get("data", {})

        # ── 1. 自定义事件 — 直接透传 ──
        #    节点通过 dispatch_custom_event(...) 发出的事件：
        #     - thought, tool_parallel_start, tool_start, tool_end, reflection
        if kind == "on_custom_event":
            payload = data.get("payload", data)
            if isinstance(payload, dict) and payload.get("type") in ALL_EVENTS:
                yield payload
            continue

        # ── 2. on_tool_start — 工具开始调用 ──
        if kind == "on_tool_start":
            tool_input = data.get("input", "")
            args = (
                tool_input
                if isinstance(tool_input, dict)
                else {"input": str(tool_input)}
            )
            yield {
                "type": EVENT_TOOL_CALL,
                "name": name,
                "args": args,
            }
            continue

        # ── 3. on_tool_end — 工具执行完成 ──
        if kind == "on_tool_end":
            output = data.get("output", "")
            if isinstance(output, dict) and "error" in output:
                yield {
                    "type": EVENT_TOOL_ERROR,
                    "name": name,
                    "tool": name,
                    "error": str(output["error"]),
                }
            else:
                yield {
                    "type": EVENT_TOOL_RESULT,
                    "name": name,
                    "result": str(output)[:3000],
                }
            continue

        # ── 4. on_chat_model_stream — 流式文本 ──
        if kind == "on_chat_model_stream":
            chunk = data.get("chunk", "")
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield {"type": EVENT_TEXT, "content": content}
            continue

        # ── 5. 节点级事件 (信息性) ──
        if kind == "on_chain_start":
            # agent/classify 节点开始时发出 thought 提示
            if name in ("agent", "classify"):
                yield {
                    "type": EVENT_THOUGHT,
                    "content": f"🔄 正在处理..." if name == "agent" else "📋 正在分析请求类型...",
                }
            continue

        # 其他事件忽略
        continue
