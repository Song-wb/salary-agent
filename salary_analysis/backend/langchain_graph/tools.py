"""LangChain 工具包装 — 将现有工具包装为 StructuredTool

从 agent/tools/ 中导入现有的 handler 函数和 JSON Schema 定义，
转换为 LangChain 兼容的 StructuredTool 列表。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

# ── 导入现有工具 handler ────────────────────────────────────────
from agent.tools.salary import query_salary_statistics, query_recruitment_list
from agent.tools.analysis import compare_cities, compare_industries
from agent.tools.mcp_tools import create_word_document
from agent.tools.memory_tools import make_search_handler

# ── 导入工具定义 (JSON Schema) ──────────────────────────────────
from agent.tools.salary import SALARY_TOOL_DEFINITIONS
from agent.tools.analysis import ANALYSIS_TOOL_DEFINITIONS
from agent.tools.mcp_tools import MCP_TOOL_DEFINITIONS


def _json_schema_to_pydantic(
    name: str, parameters: dict,
) -> type[BaseModel]:
    """将 JSON Schema 转换为 Pydantic Model (用于 StructuredTool args_schema)

    使用 pydantic.create_model 动态创建 Pydantic v2 兼容的模型。
    """
    fields: dict = {}
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for prop_name, prop_schema in properties.items():
        py_type = TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        default = prop_schema.get("default", ...)

        # pydantic.create_model 期望: (type, default_value_or_field)
        if default is ... or prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (py_type, Field(default=default, description=description))

    return create_model(f"{name}_Params", **fields)


def _build_structured_tool(
    name: str,
    description: str,
    parameters: dict,
    handler: Any,
) -> StructuredTool:
    """构建单个 StructuredTool

    自动处理 async handler，检测 handler 是否为 coroutine function。

    Pydantic v2 兼容性: StructuredTool.from_function 的 args_schema
    在 Pydantic v2 中可以直接使用动态创建的模型。
    """
    import inspect

    is_async = inspect.iscoroutinefunction(handler)

    # 构建 args_schema
    args_schema = _json_schema_to_pydantic(name, parameters)

    # 构建包装函数: 将 **kwargs 透传给 handler
    async def _async_wrapper(**kwargs: Any) -> Any:
        return await handler(**kwargs)

    def _sync_wrapper(**kwargs: Any) -> Any:
        return handler(**kwargs)

    wrapper = _async_wrapper if is_async else _sync_wrapper

    return StructuredTool.from_function(
        name=name,
        description=description,
        args_schema=args_schema,
        func=None,              # 同步 fallback (不启用)
        coroutine=wrapper,      # async 主入口
        return_direct=False,
    )


def build_agent_tools() -> list[StructuredTool]:
    """构建所有 Agent 可用的工具列表 (LangChain StructuredTool)

    从现有的 _TOOL_HANDLERS + 工具定义中收集并包装所有工具。
    """
    # 工具名 → handler 映射 (同 agent/tools/__init__.py)
    TOOL_HANDLERS = {
        "query_salary_statistics": query_salary_statistics,
        "query_recruitment_list": query_recruitment_list,
        "compare_cities": compare_cities,
        "compare_industries": compare_industries,
        "create_word_document": create_word_document,
    }

    ALL_DEFINITIONS = (
        SALARY_TOOL_DEFINITIONS
        + ANALYSIS_TOOL_DEFINITIONS
        + MCP_TOOL_DEFINITIONS
    )

    tools: list[StructuredTool] = []
    for definition in ALL_DEFINITIONS:
        name = definition["name"]
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            continue

        tool = _build_structured_tool(
            name=name,
            description=definition["description"],
            parameters=definition["parameters"],
            handler=handler,
        )
        tools.append(tool)

    return tools


def build_search_memory_tool(memory_instance) -> StructuredTool:
    """单独构建 search_memory 工具 (需要在运行时注入 memory 实例)

    与 Orchestrator 不同，这里主动接受 memory 实例而非通过闭包延迟创建。

    Args:
        memory_instance: EmbeddingMemory 或 VectorMemory 实例

    Returns:
        绑定到该 memory 实例的 StructuredTool
    """
    from agent.tools.memory_tools import SEARCH_MEMORY_TOOL

    definitions = SEARCH_MEMORY_TOOL
    handler = make_search_handler(memory_instance)

    return _build_structured_tool(
        name=definitions["name"],
        description=definitions["description"],
        parameters=definitions["parameters"],
        handler=handler,
    )


def tools_to_openai_format(tools: list[StructuredTool]) -> list[dict]:
    """将 StructuredTool 列表转为 OpenAI function calling 格式

    兼容原有 registry.get_openai_tools() 的输出格式。
    用于流式事件中将工具信息传递给前端。
    """
    result = []
    for t in tools:
        schema = t.args_schema.schema() if hasattr(t, "args_schema") else {}
        result.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema,
            },
        })
    return result
