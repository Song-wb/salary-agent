"""Supervisor 监督图 — 多 Agent 编排

替换 agent/orchestrator.py 的手动路由逻辑。

架构：
  用户输入
     │
     ▼
  Supervisor.classify() ── LLM 分类
     │
     ├── simple    → SimpleReActGraph
     ├── analysis  → AnalysisReActGraph
     ├── report    → DataCollectionGraph → ReportWriterGraph
     ├── chart     → ChartReActGraph
     └── complex   → ComplexReActGraph (大步数)
     │
     ▼
  返回最终结果
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END

from langchain_graph.state import AgentState, make_initial_state
from langchain_graph.graphs.react_graph import compile_react_agent, DEFAULT_SYSTEM_PROMPT
from langchain_graph.streaming import (
    langgraph_events_to_sse,
    create_done_event,
    create_task_type_event,
)

logger = logging.getLogger("langchain_graph.graphs.supervisor")

# ── Specialist 系统提示词 ─────────────────────────────────────

ORCHESTRATOR_PROMPT = """你是一个智能助手编排器，负责分析用户请求并决定处理策略。

对于用户请求，判断其属于以下哪种类型，只输出一个标签：

- simple: 简单的信息查询（如"北京Java开发薪资多少"），单次工具调用即可回答
- analysis: 需要多维度分析（如"对比北京和上海的互联网行业薪资"），需要多次工具调用和计算
- report: 需要生成完整报告（如"生成北京互联网行业报告"），需要结构化输出
- chart: 需要可视化图表建议
- complex: 复杂任务，需要拆分为多个子任务

输出格式：只输出标签，不要输出其他内容。"""

ANALYST_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**数据分析师**，专精于薪资数据的查询和统计分析。

你的工作方式：
1. 理解分析需求，确定需要哪些数据
2. 使用工具获取原始数据
3. 对数据进行对比、排序、趋势分析
4. 总结发现，给出数据驱动的结论

你可以使用的工具：
- query_salary_statistics: 查询单行业/城市的薪资统计
- query_recruitment_list: 查询原始招聘信息
- compare_cities: 跨城市对比
- compare_industries: 跨行业对比

分析时请注意：
- 对比时要控制变量（如对比不同城市时保持行业一致）
- 关注样本量，样本过少时给出提醒
- 不仅报告数据，还要解读数据背后的含义"""

REPORT_WRITER_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**报告撰写师**，专精于将数据组织为结构化的专业分析报告。

你的工作方式：
1. 先获取或确认数据已就绪
2. 设计报告结构（概述、分析、结论）
3. 撰写完整的 Markdown 报告

报告要求：
- 使用 Markdown 格式，包含标题层级
- 每个章节必须有数据支撑
- 包含数据解读和洞察
- 最后给出结论和建议

报告标准结构：
## 1. 市场概况
## 2. 核心数据发现
## 3. 详细分析
## 4. 趋势与洞察
## 5. 结论与建议

如果数据不足，请先使用工具获取数据再撰写报告。"""

CHART_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**图表创建师**，专精于确定最佳的数据可视化方案。

你的工作方式：
1. 理解数据和要传达的信息
2. 选择合适的图表类型
3. 描述图表的结构、数据映射和样式

图表类型选择指南：
- 薪资对比 → 柱状图（bar chart）
- 薪资分布 → 箱线图（box plot）或直方图
- 趋势变化 → 折线图（line chart）
- 占比关系 → 饼图（pie chart）或环形图
- 多维度 → 分组柱状图或热力图

输出格式：描述你推荐的图表方案，包括：
- 图表类型
- X轴/Y轴映射
- 数据来源
- 期望传达的洞察"""


class Supervisor:
    """多 Agent 监督编排器

    负责 LLM 请求分类 + 路由到 Specialist 子图 + 结果汇总。
    与原 Orchestrator 不同的是，内部使用 LangGraph 编译的子图
    而非手写 ReAct 循环。

    Usage:
        supervisor = Supervisor(llm, tools)
        result = await supervisor.process("北京Java开发薪资")
        # 或流式:
        async for event in supervisor.process_stream("北京Java开发薪资"):
            ...
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[StructuredTool],
        memory_instance=None,
    ):
        self.llm = llm
        self.tools = tools
        self.vector_memory = memory_instance

        # 子图缓存 (lazy init)
        self._sub_graphs: dict[str, Any] = {}

    # ── 子图工厂 (延迟构建) ──────────────────────────────────

    def _get_sub_graph(self, name: str, system_prompt: str, max_steps: int = 10):
        """获取或构建子图"""
        if name not in self._sub_graphs:
            # 如果有 vector_memory, 添加 search_memory 工具
            tools = list(self.tools)
            if self.vector_memory is not None:
                from langchain_graph.tools import build_search_memory_tool
                search_tool = build_search_memory_tool(self.vector_memory)
                tools.append(search_tool)

            self._sub_graphs[name] = compile_react_agent(
                llm=self.llm,
                tools=tools,
                system_prompt=system_prompt,
                max_steps=max_steps,
                with_reflection=True,
            )
        return self._sub_graphs[name]

    @property
    def simple_graph(self):
        return self._get_sub_graph("simple", DEFAULT_SYSTEM_PROMPT, max_steps=6)

    @property
    def analysis_graph(self):
        return self._get_sub_graph("analysis", ANALYST_PROMPT, max_steps=8)

    @property
    def chart_graph(self):
        return self._get_sub_graph("chart", CHART_PROMPT, max_steps=6)

    @property
    def complex_graph(self):
        return self._get_sub_graph("complex", DEFAULT_SYSTEM_PROMPT, max_steps=12)

    @property
    def data_collect_graph(self):
        return self._get_sub_graph("report_data", ANALYST_PROMPT, max_steps=6)

    @property
    def report_write_graph(self):
        return self._get_sub_graph("report_write", REPORT_WRITER_PROMPT, max_steps=4)

    # ── 任务分类 ──────────────────────────────────────────────

    async def classify(self, message: str) -> str:
        """使用 LLM 分析请求类型

        Returns:
            simple / analysis / report / chart / complex (默认 simple)
        """
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=ORCHESTRATOR_PROMPT),
                HumanMessage(content=f"用户请求：{message}"),
            ])
            task_type = response.content.strip().lower()
            valid = {"simple", "analysis", "report", "chart", "complex"}
            return task_type if task_type in valid else "simple"
        except Exception as e:
            logger.warning(f"分类失败，默认 simple: {e}")
            return "simple"

    # ── 非流式处理 ────────────────────────────────────────────

    async def process(
        self,
        message: str,
        industry: str = "互联网",
        city: str = "北京",
        short_term_summary: str = "",
        session_cache: dict | None = None,
    ) -> dict:
        """执行完整 Agent 处理

        Args:
            message: 用户消息
            industry: 行业
            city: 城市
            short_term_summary: 短期记忆摘要
            session_cache: L3 会话缓存

        Returns:
            dict with keys: final_answer, task_type, token_usage
        """
        # 1. 分类
        task_type = await self.classify(message)

        # 2. 构建初始状态
        state = make_initial_state(
            user_message=message,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_type=task_type,
            industry=industry,
            city=city,
            short_term_summary=short_term_summary,
        )
        state["messages"] = [HumanMessage(content=message)]
        if session_cache:
            state["session_cache"] = session_cache

        # 3. 路由到子图
        config = {"configurable": {"thread_id": f"sup_{state['trace_id']}"}}

        try:
            if task_type == "report":
                # 阶段1: 数据收集
                collect_state = await self.data_collect_graph.ainvoke(state, config)
                # 阶段2: 报告撰写
                final_state = await self.report_write_graph.ainvoke(collect_state, config)
            else:
                sub_graph = {
                    "simple": self.simple_graph,
                    "analysis": self.analysis_graph,
                    "chart": self.chart_graph,
                    "complex": self.complex_graph,
                }.get(task_type, self.simple_graph)
                final_state = await sub_graph.ainvoke(state, config)

        except Exception as e:
            logger.error(f"处理失败 [{task_type}]: {e}", exc_info=True)
            return {
                "final_answer": f"处理失败: {str(e)}",
                "task_type": task_type,
                "token_usage": {},
            }

        # 4. 提取最终回答
        messages = final_state.get("messages", [])
        final_answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                final_answer = msg.content
                break

        return {
            "final_answer": final_answer,
            "task_type": task_type,
            "token_usage": {},
        }

    # ── 流式处理 ──────────────────────────────────────────────

    async def process_stream(
        self,
        message: str,
        industry: str = "互联网",
        city: str = "北京",
        short_term_summary: str = "",
        session_cache: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式处理，产生 SSE 兼容事件

        先产出 task_type 事件，然后委托给对应子图的 astream_events。
        """
        # 1. 分类
        task_type = await self.classify(message)
        yield create_task_type_event(task_type)

        # 2. 构建初始状态
        state = make_initial_state(
            user_message=message,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_type=task_type,
            industry=industry,
            city=city,
            short_term_summary=short_term_summary,
        )
        state["messages"] = [HumanMessage(content=message)]
        if session_cache:
            state["session_cache"] = session_cache

        config = {"configurable": {"thread_id": f"sup_{state['trace_id']}"}}
        final_content = ""

        try:
            if task_type == "report":
                yield {"type": "thought", "content": "📋 准备生成报告，先收集数据..."}
                # 阶段1: 数据收集 (流式)
                async for event in self._stream_sub_graph(
                    self.data_collect_graph, state, config
                ):
                    yield event

                yield {"type": "thought", "content": "📝 数据已就绪，正在撰写分析报告..."}
                # 阶段2: 报告撰写 (流式)
                async for event in self._stream_sub_graph(
                    self.report_write_graph, state, config
                ):
                    if event.get("type") == "text":
                        final_content += event.get("content", "")
                    yield event
            else:
                sub_graph = {
                    "simple": self.simple_graph,
                    "analysis": self.analysis_graph,
                    "chart": self.chart_graph,
                    "complex": self.complex_graph,
                }.get(task_type, self.simple_graph)

                async for event in self._stream_sub_graph(sub_graph, state, config):
                    if event.get("type") == "text":
                        final_content += event.get("content", "")
                    yield event

        except Exception as e:
            logger.error(f"流式处理失败 [{task_type}]: {e}")
            yield {"type": "tool_error", "error": str(e)}

        # 产出 done 事件
        yield create_done_event(content=final_content)

    async def _stream_sub_graph(
        self,
        graph: Any,
        state: dict,
        config: dict,
    ) -> AsyncGenerator[dict, None]:
        """将子图的 astream_events 映射为 SSE 事件

        使用 streaming.py 的 langgraph_events_to_sse 做统一映射，
        覆盖前端消费的全部 11 种事件类型。
        """
        from langchain_graph.streaming import langgraph_events_to_sse

        lg_events = graph.astream_events(state, config=config, version="v2")
        async for sse_event in langgraph_events_to_sse(lg_events):
            yield sse_event

    # ── 工具列表 ──────────────────────────────────────────────

    def get_openai_tools(self) -> list[dict]:
        """返回 OpenAI function calling 格式的工具列表"""
        from langchain_graph.tools import tools_to_openai_format
        return tools_to_openai_format(self.tools)

    def list_tools(self) -> list[dict]:
        """返回工具名列表 (兼容原 registry.list_tools)"""
        return [{"name": t.name, "description": t.description} for t in self.tools]
