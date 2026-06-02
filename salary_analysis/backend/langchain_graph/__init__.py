"""LangChain / LangGraph 薪资分析 Agent 重构包

核心 API:
    from langchain_graph.adapter import LangGraphOrchestrator
    orchestrator = LangGraphOrchestrator()
    result = await orchestrator.process("北京Java薪资")

模块结构:
    state.py         — AgentState TypedDict 定义
    llm.py           — DeepSeek ChatModel (带熔断器)
    tools.py         — 工具 StructuredTool 包装
    graphs/
        react_graph.py      — 基础 ReAct Agent 图
        supervisor_graph.py — 多 Agent 编排监督图
    nodes/
        agent_node.py       — LLM 调用节点
        tool_node.py        — 工具执行节点 (L3缓存+并行)
        reflection_node.py  — 自纠错反射节点
    streaming.py     — LangGraph 事件 → SSE 映射
    adapter.py       — 兼容原 Orchestrator 接口的适配器
"""

from langchain_graph.adapter import LangGraphOrchestrator
from langchain_graph.graphs.react_graph import build_react_graph, compile_react_agent
from langchain_graph.graphs.supervisor_graph import Supervisor
from langchain_graph.llm import create_llm
from langchain_graph.tools import build_agent_tools

__all__ = [
    "LangGraphOrchestrator",
    "build_react_graph",
    "compile_react_agent",
    "Supervisor",
    "create_llm",
    "build_agent_tools",
]
