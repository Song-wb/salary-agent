"""LangGraph → 原 Orchestrator 接口适配器

提供与 agent.orchestrator.Orchestrator 兼容的 process / process_stream 接口，
内部使用 Supervisor + 子图 替代原有的 hand-written ReActAgent。
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import HumanMessage

from ai.config import AIConfig
from agent.memory import EmbeddingMemory, WorkingMemory, ShortTermMemory
from agent.core import AgentResult
from agent.observability import TraceContext, global_metrics

from langchain_graph.state import make_initial_state
from langchain_graph.llm import create_llm
from langchain_graph.tools import build_agent_tools
from langchain_graph.graphs.supervisor_graph import Supervisor


class LangGraphOrchestrator:
    """兼容原 Orchestrator 接口的 LangGraph 适配器

    内部使用 Supervisor + 子图 替代原手写循环。

    Usage:
        from langchain_graph.adapter import LangGraphOrchestrator
        orchestrator = LangGraphOrchestrator(config)
        # 可直接替换原有的 orchestrator = Orchestrator(config, registry)
    """

    def __init__(self, config: Optional[AIConfig] = None, registry=None):
        self.config = config or AIConfig()
        self.llm = create_llm(self.config)

        self.langchain_tools = build_agent_tools()

        # 记忆系统
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.vector = EmbeddingMemory()
        self._query_cache: dict = {}

        # Supervisor (多 Agent 编排器)
        self.supervisor = Supervisor(
            llm=self.llm,
            tools=self.langchain_tools,
            memory_instance=self.vector,
        )

        # 暴露 registry 保持接口兼容
        self.registry = registry

    async def process(self, message: str) -> AgentResult:
        """兼容原 orchestrator.process()"""
        self.short_term.add_user(message)
        self.working.set_task(message)

        # 委托给 Supervisor
        result = await self.supervisor.process(
            message=message,
            industry=self.working.current_industry,
            city=self.working.current_city,
            short_term_summary=self.short_term.get_context().get("summary", ""),
            session_cache=self._query_cache,
        )

        final_answer = result.get("final_answer", "")
        task_type = result.get("task_type", "simple")

        agent_result = AgentResult(
            final_answer=final_answer,
            steps=[],
            total_time=0.0,
        )

        self.short_term.add_assistant(final_answer)
        self._save_insight(message, agent_result)

        return agent_result

    async def process_stream(self, message: str) -> AsyncGenerator[dict, None]:
        """兼容原 orchestrator.process_stream()"""
        self.short_term.add_user(message)
        self.working.set_task(message)

        final_content = ""

        async for event in self.supervisor.process_stream(
            message=message,
            industry=self.working.current_industry,
            city=self.working.current_city,
            short_term_summary=self.short_term.get_context().get("summary", ""),
            session_cache=self._query_cache,
        ):
            if event.get("type") == "done":
                final_content = event.get("content", "")
            yield event

        if final_content:
            self.short_term.add_assistant(final_content)

    def _save_insight(self, message: str, result: AgentResult):
        """保存分析洞察到长期记忆"""
        if result.final_answer and len(result.final_answer) > 50:
            self.vector.remember(
                content=f"Q: {message}\nA: {result.final_answer[:300]}",
                tags=[
                    self.working.current_industry,
                    self.working.current_city,
                    *self.working.focus_positions,
                ],
                weight=1.0,
            )

    def get_status(self) -> dict:
        return {
            "short_term": {
                "messages": len(self.short_term.messages),
                "has_summary": bool(self.short_term.summary),
            },
            "working": self.working.to_dict(),
            "vector_memory": {
                "count": self.vector.count(),
                "path": self.vector.storage_path,
            },
            "supervisor_tools": [t.name for t in self.supervisor.tools],
        }
