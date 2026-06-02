"""流式 SSE 事件测试 — 验证所有 11 种事件类型的格式和映射"""

import json
import pytest
from typing import AsyncGenerator

from langchain_graph.streaming import (
    create_task_type_event,
    create_thought_event,
    create_tool_parallel_start_event,
    create_tool_call_event,
    create_tool_start_event,
    create_tool_end_event,
    create_tool_error_event,
    create_tool_result_event,
    create_reflection_event,
    create_text_event,
    create_done_event,
    langgraph_events_to_sse,
)


class TestEventFactories:
    """事件工厂函数测试 — 验证所有事件创建函数输出正确的 dict 格式"""

    def test_task_type_event(self):
        event = create_task_type_event("analysis")
        assert event == {"type": "task_type", "task_type": "analysis"}

    def test_thought_event(self):
        event = create_thought_event("分析中...")
        assert event == {"type": "thought", "content": "分析中..."}

    def test_tool_parallel_start_event(self):
        event = create_tool_parallel_start_event(
            2, ["query_salary_statistics", "query_recruitment_list"],
        )
        assert event["type"] == "tool_parallel_start"
        assert event["count"] == 2
        assert len(event["names"]) == 2

    def test_tool_call_event(self):
        event = create_tool_call_event(
            "query_salary_statistics",
            {"industry": "互联网", "city": "北京"},
            thought="需要查询数据",
        )
        assert event["type"] == "tool_call"
        assert event["name"] == "query_salary_statistics"
        assert event["args"]["city"] == "北京"
        assert event["thought"] == "需要查询数据"

    def test_tool_start_event(self):
        event = create_tool_start_event("query_salary_statistics", {"city": "北京"})
        assert event["type"] == "tool_start"
        assert event["tool"] == "query_salary_statistics"

    def test_tool_end_event(self):
        event = create_tool_end_event("query_salary_statistics", 1.23, "结果摘要")
        assert event["type"] == "tool_end"
        assert event["tool"] == "query_salary_statistics"
        assert event["duration_s"] == 1.23
        assert event["result_summary"] == "结果摘要"

    def test_tool_error_event(self):
        event = create_tool_error_event("query_salary_statistics", "连接超时")
        assert event["type"] == "tool_error"
        assert event["tool"] == "query_salary_statistics"
        assert "连接超时" in event["error"]

    def test_tool_result_event(self):
        event = create_tool_result_event("query_salary_statistics", '{"avg": 25000}')
        assert event["type"] == "tool_result"
        assert event["name"] == "query_salary_statistics"

    def test_reflection_event(self):
        issues = [{"type": "empty_result", "severity": "high", "message": "空结果"}]
        event = create_reflection_event(issues, reflection_depth=1, max_depth=2)
        assert event["type"] == "reflection"
        assert len(event["issues"]) == 1
        assert event["reflection_depth"] == 1
        assert event["max_depth"] == 2

    def test_text_event(self):
        event = create_text_event("北京平均薪资")
        assert event == {"type": "text", "content": "北京平均薪资"}

    def test_done_event(self):
        event = create_done_event(
            content="完成",
            steps=3,
            token_usage={"total_tokens": 100},
            total_time=5.0,
        )
        assert event["type"] == "done"
        assert event["content"] == "完成"
        assert event["steps"] == 3
        assert event["token_usage"]["total_tokens"] == 100
        assert event["total_time"] == 5.0


class TestLangGraphToSSEMapping:
    """langgraph_events_to_sse 映射测试 — 验证 LangGraph 事件被正确转换为 SSE 格式"""

    @staticmethod
    async def _mock_lg_events(events: list[dict]) -> AsyncGenerator[dict, None]:
        """辅助: 将事件列表转为异步生成器"""
        for event in events:
            yield event

    async def _collect(self, events: list[dict]) -> list[dict]:
        """辅助: 将 mock LangGraph 事件通过映射器收集为 SSE 事件列表"""
        gen = self._mock_lg_events(events)
        results = []
        async for sse_event in langgraph_events_to_sse(gen):
            results.append(sse_event)
        return results

    @pytest.mark.asyncio
    async def test_custom_event_passthrough(self):
        """自定义事件直接透传"""
        lg_events = [
            {"event": "on_custom_event", "data": {
                "payload": {"type": "thought", "content": "分析中"},
            }},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "thought"
        assert sse_events[0]["content"] == "分析中"

    @pytest.mark.asyncio
    async def test_custom_event_unknown_type_filtered(self):
        """不在 ALL_EVENTS 中的自定义事件类型被过滤"""
        lg_events = [
            {"event": "on_custom_event", "data": {
                "payload": {"type": "internal_debug", "data": "x"},
            }},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 0

    @pytest.mark.asyncio
    async def test_tool_start_maps_to_tool_call(self):
        """on_tool_start → tool_call"""
        lg_events = [
            {"event": "on_tool_start", "name": "query_salary_statistics",
             "data": {"input": {"industry": "互联网", "city": "北京"}}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "tool_call"
        assert sse_events[0]["name"] == "query_salary_statistics"

    @pytest.mark.asyncio
    async def test_tool_end_maps_to_tool_result(self):
        """on_tool_end → tool_result (正常)"""
        lg_events = [
            {"event": "on_tool_end", "name": "query_salary_statistics",
             "data": {"output": '{"avg_salary": 25000}'}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "tool_result"

    @pytest.mark.asyncio
    async def test_tool_end_error_maps_to_tool_error(self):
        """on_tool_end (含 error) → tool_error"""
        lg_events = [
            {"event": "on_tool_end", "name": "query_salary_statistics",
             "data": {"output": {"error": "连接失败"}}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "tool_error"
        assert "连接失败" in sse_events[0]["error"]

    @pytest.mark.asyncio
    async def test_chat_model_stream_maps_to_text(self):
        """on_chat_model_stream → text"""
        class MockChunk:
            content = "北京"

        lg_events = [
            {"event": "on_chat_model_stream", "name": "ChatOpenAI",
             "data": {"chunk": MockChunk()}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "text"
        assert sse_events[0]["content"] == "北京"

    @pytest.mark.asyncio
    async def test_chain_start_maps_to_thought(self):
        """on_chain_start (agent/classify) → thought"""
        lg_events = [
            {"event": "on_chain_start", "name": "agent",
             "data": {}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 1
        assert sse_events[0]["type"] == "thought"

    @pytest.mark.asyncio
    async def test_full_event_sequence(self):
        """完整的典型 Agent 执行事件序列"""
        class MockChunk:
            content = "正在分析"

        lg_events = [
            # 1. 节点开始
            {"event": "on_chain_start", "name": "agent", "data": {}},
            # 2. LLM 流式输出
            {"event": "on_chat_model_stream", "name": "ChatOpenAI",
             "data": {"chunk": MockChunk()}},
            # 3. 工具开始
            {"event": "on_tool_start", "name": "query_salary_statistics",
             "data": {"input": {"industry": "互联网", "city": "北京"}}},
            # 4. 自定义: 工具开始 (并行信息)
            {"event": "on_custom_event", "data": {
                "payload": {"type": "tool_parallel_start",
                            "count": 1, "names": ["query_salary_statistics"]},
            }},
            # 5. 工具结束
            {"event": "on_tool_end", "name": "query_salary_statistics",
             "data": {"output": '{"avg_salary": 25000}'}},
            # 6. 自定义: tool_end
            {"event": "on_custom_event", "data": {
                "payload": {"type": "tool_end", "tool": "query_salary_statistics",
                            "duration_s": 0.5, "result_summary": "25000"},
            }},
        ]

        sse_events = await self._collect(lg_events)
        types = [e["type"] for e in sse_events]

        assert types == [
            "thought",           # agent 节点开始
            "text",              # LLM 流式输出
            "tool_call",         # 工具开始
            "tool_parallel_start",  # 自定义并行
            "tool_result",       # 工具结果
            "tool_end",          # 自定义结束
        ]

    @pytest.mark.asyncio
    async def test_unknown_events_ignored(self):
        """不关心的事件类型被忽略"""
        lg_events = [
            {"event": "on_debug", "data": {}},
            {"event": "on_chain_end", "data": {}},
            {"event": "on_parser_start", "data": {}},
        ]
        sse_events = await self._collect(lg_events)
        assert len(sse_events) == 0


class TestDispatchModule:
    """dispatch_custom_event 封装测试 — 验证 emit_event 不崩溃"""

    @pytest.mark.asyncio
    async def test_emit_event_basic(self):
        """发射事件不抛出异常"""
        from langchain_graph._dispatch import emit_event
        await emit_event({"type": "thought", "content": "test"})

    @pytest.mark.asyncio
    async def test_emit_event_missing_type(self):
        """缺少 type 字段静默忽略"""
        from langchain_graph._dispatch import emit_event
        await emit_event({"content": "no type"})

    @pytest.mark.asyncio
    async def test_emit_event_all_types(self):
        """所有 11 种事件类型都能正常发射"""
        from langchain_graph._dispatch import emit_event
        factories = [
            ("task_type", lambda: {"type": "task_type", "task_type": "simple"}),
            ("thought", lambda: {"type": "thought", "content": "test"}),
            ("tool_parallel_start", lambda: {"type": "tool_parallel_start", "count": 1, "names": ["t"]}),
            ("tool_call", lambda: {"type": "tool_call", "name": "t", "args": {}}),
            ("tool_start", lambda: {"type": "tool_start", "tool": "t", "args": {}}),
            ("tool_end", lambda: {"type": "tool_end", "tool": "t", "duration_s": 0.5}),
            ("tool_error", lambda: {"type": "tool_error", "tool": "t", "error": "e"}),
            ("tool_result", lambda: {"type": "tool_result", "name": "t", "result": "r"}),
            ("reflection", lambda: {"type": "reflection", "issues": [], "reflection_depth": 1, "max_depth": 2}),
            ("text", lambda: {"type": "text", "content": "t"}),
            ("done", lambda: {"type": "done", "content": "t", "steps": 1}),
        ]
        for name, factory in factories:
            await emit_event(factory())
