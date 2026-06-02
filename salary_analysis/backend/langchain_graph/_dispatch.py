"""自定义事件发射工具 — 将 LangGraph dispatch_custom_event 统一封装

LangGraph 的 astream_events 支持 on_custom_event 事件类型。
节点通过 dispatch_custom_event() 发出的事件会以 on_custom_event 形式
出现在 astream_events 流中，进而被 streaming.py 转换为 SSE 事件。

用法:
    from langchain_graph._dispatch import emit_event
    await emit_event({"type": "thought", "content": "..."})
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("langchain_graph.dispatch")

try:
    from langgraph.graph import dispatch_custom_event as _dispatch
except ImportError:
    # 降级: langgraph 版本过低时不发射自定义事件
    logger.info("langgraph.graph.dispatch_custom_event 不可用，自定义事件将被静默忽略")

    async def _dispatch(name: str, data: dict) -> None:
        pass


async def emit_event(payload: dict) -> None:
    """发射自定义事件 (透传给前端的 SSE 流)

    Args:
        payload: 事件字典，必须包含 "type" 字段。
                所有字段会原样出现在 astream_events 的
                on_custom_event → data → payload 中。

    前端消费的事件类型:
        thought, tool_parallel_start, tool_start,
        tool_end, reflection 等。
    """
    if "type" not in payload:
        logger.warning(f"自定义事件缺少 type 字段: {list(payload.keys())}")
        return

    name = payload["type"]
    await _dispatch(name, payload)
