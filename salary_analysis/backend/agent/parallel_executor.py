"""并行工具执行器 — 限流、超时、事件发射

整合方案一（工具内部并行）和方案二（ReAct 循环并行）的能力，
为方案三（全异步流式 + 逐步渲染）提供基础设施。

核心组件：
- ParallelThrottle: 基于 asyncio.Semaphore 的限流器，控制最大并发数
- ToolExecutionEvent: 工具执行事件，记录每次调用的完整生命周期
- ParallelExecutor: 整合限流 + 超时的并行执行器，支持 batch 和 stream 模式
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator

logger = logging.getLogger("agent.executor")


class ParallelThrottle:
    """信号量限流器 — 控制最大并发数，防止后端过载

    使用场景：
    - 对比 5 个城市时，限制最多 3 个并发 HTTP 请求
    - 多用户同时使用 Agent 时，避免 MySQL 连接池耗尽

    原理：
    基于 asyncio.Semaphore(N)，超过 N 个并发时后续协程排队等待，
    直到前序协程释放信号量。
    """

    def __init__(self, max_concurrent: int = 3, name: str = "default"):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._name = name
        self._active = 0
        self._max = max_concurrent

    @property
    def active(self) -> int:
        return self._active

    @property
    def available(self) -> int:
        return self._max - self._active

    async def execute(self, tool_name: str, fn, **kwargs):
        """限流执行：超过并发上限则排队等待"""
        async with self._semaphore:
            self._active += 1
            try:
                return await fn(**kwargs)
            finally:
                self._active -= 1


class ToolExecutionEvent:
    """工具执行事件 — 记录单次工具调用的完整生命周期

    用于：
    1. 追踪每次工具调用的耗时
    2. 向前端 SSE 推送实时状态
    3. 记录执行结果或错误
    """

    def __init__(self, tool_name: str, args: dict):
        self.tool_name = tool_name
        self.args = args
        self.start_time = 0.0
        self.end_time = 0.0
        self.result = None
        self.error = None

    @property
    def duration(self) -> float:
        if self.end_time and self.start_time:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    @property
    def success(self) -> bool:
        return self.error is None and self.result is not None

    def start(self):
        self.start_time = time.time()
        return self

    def finish(self, result):
        self.result = result
        self.end_time = time.time()
        return self

    def fail(self, error: str):
        self.error = error
        self.end_time = time.time()
        return self

    def to_start_event(self) -> dict:
        """工具开始执行事件（SSE 推送）"""
        return {
            "type": "tool_start",
            "tool": self.tool_name,
            "args": self.args,
        }

    def to_end_event(self) -> dict:
        """工具执行完毕事件（SSE 推送）"""
        return {
            "type": "tool_end",
            "tool": self.tool_name,
            "duration_s": self.duration,
            "success": self.success,
            "result_summary": str(self.result)[:200] if self.success else "",
        }

    def to_error_event(self) -> dict:
        """工具执行出错事件（SSE 推送）"""
        return {
            "type": "tool_error",
            "tool": self.tool_name,
            "args": self.args,
            "error": self.error,
            "duration_s": self.duration,
        }


class ParallelExecutor:
    """并行工具执行器 — 整合限流、超时、事件发射

    提供三种执行模式：
    1. execute_single: 单工具执行（带限流 + 超时）
    2. execute_batch: 批处理并行，全部完成后一次性返回
    3. execute_batch_stream: 流式并行，每完成一个立即 yield 事件

    限流策略：
    - 默认 max_concurrent=3，避免 MySQL 连接池被打满
    - 所有执行通过 asyncio.Semaphore 控制

    超时策略：
    - 默认 15s 超时，防止个别慢查询拖慢整体
    - 超时后返回错误事件，不影响其他并行工具
    """

    def __init__(self, registry, max_concurrent: int = 3, default_timeout: float = 15.0):
        self.registry = registry
        self.throttle = ParallelThrottle(max_concurrent=max_concurrent, name="tool_executor")
        self.default_timeout = default_timeout

    async def execute_single(self, tool_name: str, args: dict) -> ToolExecutionEvent:
        """执行单个工具（带限流和超时），返回事件对象"""
        event = ToolExecutionEvent(tool_name, args).start()

        try:
            result = await self.throttle.execute(
                tool_name,
                self._run_with_timeout,
                tool_name, args,
            )
            event.finish(result)
        except asyncio.TimeoutError:
            event.fail(f"工具 [{tool_name}] 执行超时 (> {self.default_timeout}s)")
        except Exception as e:
            event.fail(f"工具 [{tool_name}] 执行失败: {str(e)}")

        return event

    async def _run_with_timeout(self, tool_name: str, args: dict):
        """带超时的工具执行"""
        return await asyncio.wait_for(
            self.registry.execute(tool_name, args),
            timeout=self.default_timeout,
        )

    async def execute_batch(self, tool_calls: list) -> list[ToolExecutionEvent]:
        """并行执行一批独立工具调用，全部完成后返回

        适合非流式场景（如 /api/agent/chat），调用方需要所有结果后再处理。
        """
        tasks = [self.execute_single(
            tc.function.name,
            self._parse_args(tc),
        ) for tc in tool_calls]

        return await asyncio.gather(*tasks)

    async def execute_batch_stream(
        self, tool_calls: list,
    ) -> AsyncGenerator[dict, None]:
        """并行执行工具，每完成一个就 yield 事件

        使用 asyncio.as_completed 实现"先完成先推送"。
        注意：工具实际是同时启动的，但工具完成事件按完成先后顺序推送。
        """
        async def run_one(tc):
            args = self._parse_args(tc)
            event = await self.execute_single(tc.function.name, args)
            return tc.id, event

        for coro in asyncio.as_completed([run_one(tc) for tc in tool_calls]):
            tc_id, event = await coro
            if event.error:
                yield event.to_error_event()
            else:
                yield event.to_end_event()

    @staticmethod
    def _parse_args(tc) -> dict:
        """解析 tool_call 的参数 JSON"""
        try:
            return json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            return {"raw": tc.function.arguments}
