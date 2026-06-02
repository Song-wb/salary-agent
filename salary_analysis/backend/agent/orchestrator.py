"""多智能体编排器 — 任务分解、Agent 调度、结果汇总"""

from typing import AsyncGenerator
from openai import AsyncOpenAI

from .core import ReActAgent, AgentResult, DEFAULT_SYSTEM_PROMPT
from .registry import ToolRegistry
from .memory import ShortTermMemory, WorkingMemory, EmbeddingMemory
from .specialists import (
    DataAnalystAgent,
    ReportWriterAgent,
    ChartCreatorAgent,
)
from .tools.memory_tools import SEARCH_MEMORY_TOOL, make_search_handler

ORCHESTRATOR_PROMPT = """你是一个智能助手编排器，负责分析用户请求并决定处理策略。

对于用户请求，判断其属于以下哪种类型，只输出一个标签：

- simple: 简单的信息查询（如"北京Java开发薪资多少"），单次工具调用即可回答
- analysis: 需要多维度分析（如"对比北京和上海的互联网行业薪资"），需要多次工具调用和计算
- report: 需要生成完整报告（如"生成北京互联网行业报告"），需要结构化输出
- chart: 需要可视化图表建议
- complex: 复杂任务，需要拆分为多个子任务

输出格式：只输出标签，不要输出其他内容。"""


class Orchestrator:
    """多 Agent 编排器

    负责：
    1. 分析用户请求复杂度
    2. 简单请求 → 直接调用 ReActAgent
    3. 复杂请求 → 分解子任务 → 分派 Specialist Agent → 汇总结果
    """

    def __init__(self, config, registry: ToolRegistry):
        self.config = config
        self.registry = registry
        self.short_term = ShortTermMemory()
        self.working = WorkingMemory()
        self.vector = EmbeddingMemory()

        # ── L3: 会话级缓存 ──
        # key: (industry, city) → dict
        # 存活期：整个 Orchestrator 生命周期（一个对话）
        self._query_cache: dict[tuple[str, str], dict] = {}

        # 注册 Agentic RAG 记忆检索工具（捕获 self.vector 引用）
        search_handler = make_search_handler(self.vector)
        registry.register_func(
            name=SEARCH_MEMORY_TOOL["name"],
            description=SEARCH_MEMORY_TOOL["description"],
            parameters=SEARCH_MEMORY_TOOL["parameters"],
            handler=search_handler,
        )

        # 主 Agent
        self.main_agent = ReActAgent(config, registry)

        # Specialist Agents
        self.data_analyst = DataAnalystAgent(config, registry)
        self.report_writer = ReportWriterAgent(config, registry)
        self.chart_creator = ChartCreatorAgent(config, registry)

        # 分类用 client
        self._classifier: AsyncOpenAI | None = None

    @property
    def classifier(self) -> AsyncOpenAI:
        if self._classifier is None:
            self._classifier = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=15,
            )
        return self._classifier

    async def process(self, message: str) -> AgentResult:
        """处理用户消息，自动判断策略"""
        self.short_term.add_user(message)

        # 1. 分析请求类型
        task_type = await self._classify(message)
        self.working.set_task(message)

        # 2. 按类型处理
        context_str = self._build_context(user_message=message)

        if task_type == "simple":
            result = await self.main_agent.run(
                user_message=message,
                context_text=context_str,
                session_cache=self._query_cache,
            )

        elif task_type == "analysis":
            result = await self._handle_analysis(message, context_str)

        elif task_type == "report":
            result = await self._handle_report(message, context_str)

        elif task_type == "chart":
            result = await self._handle_chart(message, context_str)

        else:  # complex
            result = await self._handle_complex(message, context_str)

        # 3. 保存到记忆
        self.short_term.add_assistant(result.final_answer)
        self._save_insight(message, result)

        return result

    async def process_stream(self, message: str) -> AsyncGenerator[dict, None]:
        """流式处理用户消息 — 所有任务类型都走流式事件"""
        self.short_term.add_user(message)
        task_type = await self._classify(message)
        self.working.set_task(message)
        context_str = self._build_context(user_message=message)

        yield {"type": "task_type", "task_type": task_type}

        steps = 0
        final_answer = ""

        if task_type == "simple":
            async for event in self.main_agent.run_stream(
                user_message=message,
                context_text=context_str,
                session_cache=self._query_cache,
            ):
                if event.get("type") == "done":
                    steps = event.get("steps", 0)
                    final_answer = event.get("content", "")
                yield event

        elif task_type == "analysis":
            yield {"type": "thought", "content": "🔍 收到分析请求，正在多维度查询数据..."}
            async for event in self.main_agent.run_stream(
                user_message=message,
                context_text=context_str,
                session_cache=self._query_cache,
            ):
                if event.get("type") == "done":
                    steps = event.get("steps", 0)
                    final_answer = event.get("content", "")
                yield event

        elif task_type == "report":
            yield {"type": "thought", "content": "📋 准备生成报告，先收集数据..."}
            # 阶段1：收集数据（流式执行工具）
            data_prompt = (
                f"请收集{self.working.current_industry}行业"
                f"{self.working.current_city}地区的薪资数据，用于撰写报告"
            )
            async for event in self.main_agent.run_stream(
                user_message=data_prompt,
                context_text=context_str,
                max_steps=6,
                session_cache=self._query_cache,
            ):
                # 工具调用事件透传，text/done 拦截（数据阶段不输出最终文本）
                if event["type"] in ("tool_call", "tool_result", "tool_parallel_start",
                                     "tool_start", "tool_end", "tool_error",
                                     "thought", "start"):
                    yield event

            # 阶段2：生成报告
            yield {"type": "thought", "content": "📝 数据已就绪，正在撰写分析报告..."}
            result = await self.report_writer.write(message, context_str,
                                                     session_cache=self._query_cache)
            final_answer = result.final_answer
            steps = result.tool_call_count
            yield {"type": "text", "content": final_answer}
            yield {"type": "done", "content": final_answer, "steps": steps}

        elif task_type == "chart":
            yield {"type": "thought", "content": "📊 分析数据并设计可视化方案..."}
            async for event in self.main_agent.run_stream(
                user_message=message,
                context_text=context_str,
                session_cache=self._query_cache,
            ):
                if event.get("type") == "done":
                    steps = event.get("steps", 0)
                    final_answer = event.get("content", "")
                yield event

        else:  # complex
            yield {"type": "thought", "content": "⚡ 检测到复杂任务，正在分解和执行..."}
            async for event in self.main_agent.run_stream(
                user_message=message,
                context_text=context_str,
                max_steps=12,
                session_cache=self._query_cache,
            ):
                if event.get("type") == "done":
                    steps = event.get("steps", 0)
                    final_answer = event.get("content", "")
                yield event

        self.short_term.add_assistant(final_answer if final_answer else message)
        if final_answer:
            # 构造一个临时 AgentResult 用于保存洞察
            from .core import AgentResult
            self._save_insight(message, AgentResult(final_answer=final_answer))

    async def _handle_analysis(self, message: str, context_str: str) -> AgentResult:
        """处理分析类任务"""
        result = await self.data_analyst.analyze(message, context_str,
                                                  session_cache=self._query_cache)
        return result

    async def _handle_report(self, message: str, context_str: str) -> AgentResult:
        """处理报告类任务"""
        # 先获取数据
        data_prompt = f"请收集{self.working.current_industry}行业{self.working.current_city}地区的薪资数据，用于撰写报告"
        await self.data_analyst.analyze(data_prompt, context_str,
                                         session_cache=self._query_cache)
        # 再写报告
        result = await self.report_writer.write(message, context_str,
                                                 session_cache=self._query_cache)
        return result

    async def _handle_chart(self, message: str, context_str: str) -> AgentResult:
        """处理图表类任务"""
        result = await self.chart_creator.design(message, "", context_str,
                                                  session_cache=self._query_cache)
        return result

    async def _handle_complex(self, message: str, context_str: str) -> AgentResult:
        """处理复杂任务：分解 → 执行 → 汇总"""
        # 使用主 Agent 处理复杂任务（自带 ReAct 循环和工具调用）
        result = await self.main_agent.run(
            user_message=message,
            context_text=context_str,
            max_steps=12,
            session_cache=self._query_cache,
        )
        return result

    async def _classify(self, message: str) -> str:
        """分析请求类型"""
        try:
            resp = await self.classifier.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_PROMPT},
                    {"role": "user", "content": f"用户请求：{message}"},
                ],
                temperature=0.1,
                max_tokens=20,
            )
            task_type = resp.choices[0].message.content.strip().lower()
            valid = {"simple", "analysis", "report", "chart", "complex"}
            return task_type if task_type in valid else "simple"
        except Exception:
            return "simple"

    def _build_context(self, user_message: str = "") -> str:
        """组装当前上下文

        注意：长期记忆的检索已交给 Agent 自主控制（search_memory 工具）。
        这里只组装短期记忆摘要和工作记忆。
        """
        parts = []
        # 短期记忆摘要
        st = self.short_term.get_context()
        if st["summary"]:
            parts.append(st["summary"])
        # 工作记忆
        parts.append(self.working.get_summary())
        return "\n".join(parts)

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
        """获取当前 Orchestrator 状态"""
        return {
            "short_term": {"messages": len(self.short_term.messages),
                           "has_summary": bool(self.short_term.summary)},
            "working": self.working.to_dict(),
            "vector_memory": {"count": self.vector.count(),
                              "path": self.vector.storage_path},
        }
