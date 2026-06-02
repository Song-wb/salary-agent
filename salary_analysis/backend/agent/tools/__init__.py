"""工具注册 — 将所有工具注册到 ToolRegistry"""

from ..registry import ToolRegistry
from .salary import (
    query_salary_statistics,
    query_recruitment_list,
    SALARY_TOOL_DEFINITIONS,
)
from .analysis import (
    compare_cities,
    compare_industries,
    ANALYSIS_TOOL_DEFINITIONS,
)
from .mcp_tools import (
    create_word_document,
    MCP_TOOL_DEFINITIONS,
)

# 工具名 → 处理函数映射
_TOOL_HANDLERS = {
    "query_salary_statistics": query_salary_statistics,
    "query_recruitment_list": query_recruitment_list,
    "compare_cities": compare_cities,
    "compare_industries": compare_industries,
    "create_word_document": create_word_document,
}

_ALL_TOOL_DEFINITIONS = (
    SALARY_TOOL_DEFINITIONS + ANALYSIS_TOOL_DEFINITIONS + MCP_TOOL_DEFINITIONS
)


def build_agent_tools() -> ToolRegistry:
    """构建并返回完整的 Agent 工具注册表"""
    registry = ToolRegistry()
    for definition in _ALL_TOOL_DEFINITIONS:
        name = definition["name"]
        handler = _TOOL_HANDLERS.get(name)
        if handler:
            registry.register_func(
                name=name,
                description=definition["description"],
                parameters=definition["parameters"],
                handler=handler,
            )
    return registry


__all__ = ["build_agent_tools"]
