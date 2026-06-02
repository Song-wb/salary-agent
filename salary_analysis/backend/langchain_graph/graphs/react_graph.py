"""ReAct Agent 图 — LangGraph 实现

替换 agent/core.py 中的手写 ReAct 循环。

图结构:
  START → agent_node
    │
    ├─ (tool_calls) → tool_node → reflection_node
    │                                   │
    │                        ┌── (pass) → END
    │                        └── (fail, count<max) → agent_node
    │                        └── (fail, count>=max) → END
    │
    └─ (no tool_calls) → reflection_node
                                        │
                                ┌── (pass) → END
                                └── (fail, count<max) → agent_node
                                └── (fail, count>=max) → END
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_graph.state import AgentState, make_initial_state
from langchain_graph.nodes.agent_node import create_agent_node
from langchain_graph.nodes.tool_node import create_tool_node
from langchain_graph.nodes.reflection_node import create_reflection_node

logger = logging.getLogger("langchain_graph.graphs.react_graph")

# 默认配置
DEFAULT_SYSTEM_PROMPT = """你是一个专业的薪酬分析智能助手，精通中国各行业薪酬体系。
你需要通过调用工具来获取数据，然后基于数据进行分析和回答。

你可以使用的工具包括：
- query_salary_statistics / query_recruitment_list: 查询实时薪资数据
- compare_cities / compare_industries: 多维度对比分析
- search_memory: 搜索历史分析记录，当用户问题涉及之前讨论过的内容时使用
- create_word_document: 生成 Word 分析报告

核心原则：
1. 始终基于数据说话，不得编造数据
2. 分析要有深度，提供洞察而不仅仅是罗列数字
3. 使用中文回答，薪资单位使用"元/月"
4. 回答结构清晰，包含具体数字和对比分析
5. 当用户问题与历史对话相关时，先调用 search_memory 查找背景信息
6. 获取数据后再进行分析和总结"""


def _count_tool_calls(messages: list) -> int:
    """统计消息列表中 ToolMessage 的数量 (即已执行的工具调用次数)"""
    return sum(1 for m in messages if isinstance(m, ToolMessage))


def _has_tool_calls(state: AgentState) -> str:
    """条件边: 检测 LLM 输出是否有 tool_calls

    返回 "tools"(继续执行工具) 或 "reflect"(跳过工具，直接反射检查).
    """
    messages = state.get("messages", [])
    if not messages:
        return "reflect"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "reflect"


def _check_max_steps(state: AgentState, max_steps: int = 10) -> bool:
    """检查是否达到最大步骤限制"""
    return _count_tool_calls(state.get("messages", [])) >= max_steps


def build_react_graph(
    llm: BaseChatModel,
    tools: list[StructuredTool],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_steps: int = 10,
    with_reflection: bool = True,
) -> StateGraph:
    """构建 ReAct Agent 图

    Args:
        llm: 已配置的 ChatModel 实例 (带熔断保护)
        tools: LangChain StructuredTool 列表
        system_prompt: Agent 角色系统提示词
        max_steps: 最大工具调用步骤数
        with_reflection: 是否启用反射自纠错

    Returns:
        未编译的 StateGraph (调用者选择是否编译/添加 checkpointer)
    """
    # 创建节点 (工厂函数注入依赖)
    agent_node = create_agent_node(llm)
    tool_node = create_tool_node(tools)
    reflection_node = create_reflection_node()

    # 构建图
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    if with_reflection:
        builder.add_node("reflect", reflection_node)

    # ── 边定义 ──

    # 1. 开始 → Agent
    builder.add_edge(START, "agent")

    # 2. Agent → 条件分支: tool_calls → tools, 否则 → reflect/end
    def after_agent(state: AgentState) -> str:
        step_count = _count_tool_calls(state.get("messages", []))
        decision = _has_tool_calls(state)

        if decision == "tools":
            if step_count >= max_steps:
                logger.warning(f"达到最大步骤数 ({max_steps})，强制结束工具调用")
                return "reflect" if with_reflection else END
            return "tools"

        return "reflect" if with_reflection else END

    builder.add_conditional_edges(
        "agent",
        after_agent,
        {
            "tools": "tools",
            "reflect": "reflect" if with_reflection else END,
            END: END,
        },
    )

    if with_reflection:
        # 3. Tools → Reflect
        builder.add_edge("tools", "reflect")

        # 4. Reflect → 条件分支: 通过 → end, 失败且未达上限 → agent, 失败且达上限 → end
        def after_reflect(state: AgentState) -> str:
            if state.get("needs_reflection", False):
                step_count = _count_tool_calls(state.get("messages", []))
                if step_count >= max_steps:
                    logger.warning("反射后达到最大步骤数，结束")
                    return END
                return "agent"
            return END

        builder.add_conditional_edges(
            "reflect",
            after_reflect,
            {"agent": "agent", END: END},
        )
    else:
        # 无反射: Tools → End
        builder.add_edge("tools", END)

    return builder


def build_checkpointer(
    persist_path: str | None = None,
) -> Any:
    """构建 Checkpointer (检查点/持久化)

    Args:
        persist_path: SQLite 持久化路径。
            提供时使用 SqliteSaver 实现跨会话持久化；
            为 None 时使用 MemorySaver (内存，仅当前会话)。

    Returns:
        LangGraph 检查点实例 (支持 MemorySaver 或 SqliteSaver 接口)
    """
    if persist_path is not None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3

            conn = sqlite3.connect(persist_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"使用 SQLite 持久化 Checkpointer: {persist_path}")
            return checkpointer
        except ImportError:
            logger.warning(
                "SqliteSaver 不可用 (langgraph 版本过低)，"
                "回退到 MemorySaver"
            )
        except Exception as e:
            logger.warning(f"SQLite 初始化失败 ({e})，回退到 MemorySaver")

    return MemorySaver()


def compile_react_agent(
    llm: BaseChatModel,
    tools: list[StructuredTool],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_steps: int = 10,
    with_reflection: bool = True,
    persist_path: str | None = None,
) -> Any:
    """编译 ReAct Agent 为可执行图

    便捷方法: 构建 + 编译一步完成。

    Args:
        llm: 已配置的 ChatModel
        tools: StructuredTool 列表
        system_prompt: 系统提示词
        max_steps: 最大步骤数
        with_reflection: 是否启用反射
        persist_path: SQLite 持久化路径 (None=内存)

    Returns:
        编译后的图 (支持 ainvoke / astream_events)
    """
    builder = build_react_graph(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        max_steps=max_steps,
        with_reflection=with_reflection,
    )
    checkpointer = build_checkpointer(persist_path)
    return builder.compile(checkpointer=checkpointer)
