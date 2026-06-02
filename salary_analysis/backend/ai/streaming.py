"""SSE 流式响应处理"""

from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator


async def sse_stream(generator: AsyncGenerator[str, None]):
    """将异步生成器包装为 SSE 事件流"""
    async for chunk in generator:
        yield {"data": chunk}
    yield {"data": "[DONE]"}
