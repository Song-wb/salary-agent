"""工具注册表 — 管理 Agent 可调用的所有工具"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable  # 工具实现函数
    is_async: bool = False


class ToolRegistry:
    """工具注册表：注册、发现、执行工具"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册一个工具"""
        self._tools[tool.name] = tool

    def register_func(self, name: str, description: str, parameters: dict,
                      handler: Callable):
        """快捷方式：直接注册一个工具函数"""
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_async=inspect.iscoroutinefunction(handler),
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """返回工具列表（用于展示）"""
        return [
            {"name": t.name, "description": t.description,
             "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def get_openai_tools(self) -> list[dict]:
        """返回 OpenAI-compatible function calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    @staticmethod
    def _validate_params(parameters: dict, args: dict) -> tuple[dict, dict | None]:
        """校验参数与 JSON Schema 是否匹配（白名单校验）

        Returns:
            (validated_args, None) — 校验通过
            (args, error_dict) — 校验失败，返回结构化错误信息
        """
        properties = parameters.get("properties", {})
        if not properties:
            return args, None  # 无 schema 定义，跳过校验

        issues = []
        cleaned = {}

        # 1. 填充默认值
        for key, prop in properties.items():
            if key not in args and "default" in prop:
                cleaned[key] = prop["default"]

        # 2. 逐参数校验
        for key, value in args.items():
            if key == "raw":  # JSON 解析失败时带 raw 标记，跳过校验
                cleaned[key] = value
                continue

            # 2a. 未知参数检查（白名单核心）
            if key not in properties:
                known_params = list(properties.keys())
                issues.append({
                    "type": "unknown_parameter",
                    "parameter": key,
                    "value": str(value)[:100],
                    "message": f"未知参数 '{key}'，该工具支持的参数: {known_params}",
                })
                continue

            # 2b. 类型检查
            expected_type = properties[key].get("type")
            type_ok = True
            if expected_type == "string" and not isinstance(value, str):
                type_ok = False
            elif expected_type == "array" and not isinstance(value, list):
                type_ok = False
            elif expected_type == "integer" and not isinstance(value, int):
                type_ok = False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                type_ok = False
            elif expected_type == "boolean" and not isinstance(value, bool):
                type_ok = False

            if not type_ok:
                issues.append({
                    "type": "type_mismatch",
                    "parameter": key,
                    "expected": expected_type,
                    "received": type(value).__name__,
                    "value": str(value)[:100],
                    "message": f"参数 '{key}' 类型错误: 期望 {expected_type}, "
                              f"收到 {type(value).__name__}",
                })
                continue

            # 2c. 枚举值检查（从 description 的 "可选：" 后缀提取）
            desc = properties[key].get("description", "")
            if "可选：" in desc and isinstance(value, str):
                valid_options = [
                    opt.strip() for opt in desc.split("可选：")[1]
                    .replace("，", ",").split(",")
                    if opt.strip()
                ]
                if valid_options and value not in valid_options:
                    issues.append({
                        "type": "invalid_enum_value",
                        "parameter": key,
                        "value": value,
                        "valid_values": valid_options,
                        "message": f"参数 '{key}' 的值 '{value}' 不在可选范围内: {valid_options}",
                    })
                    continue

            cleaned[key] = value

        # 3. 必需参数检查
        required = parameters.get("required", [])
        for req in required:
            if req not in args and req not in cleaned:
                issues.append({
                    "type": "missing_required",
                    "parameter": req,
                    "message": f"缺少必需参数 '{req}'",
                    "required_by": required,
                })

        if issues:
            return cleaned, {
                "error": "工具参数校验失败",
                "tool": None,  # 由 execute() 填充
                "issues": issues,
                "hint": "请根据参数说明使用正确的参数名、类型和取值范围重新调用",
            }
        return cleaned, None

    async def execute(self, name: str, args: dict) -> Any:
        """执行工具并返回结果"""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"未知工具: {name}"}

        if not isinstance(args, dict):
            return {"error": f"工具 [{name}] 参数格式错误: 期望 dict, 收到 {type(args).__name__}"}

        # 参数白名单校验
        validated_args, error = self._validate_params(tool.parameters, args)
        if error:
            error["tool"] = name
            return error

        try:
            if tool.is_async:
                return await tool.handler(**validated_args)
            return tool.handler(**validated_args)
        except Exception as e:
            return {"error": f"工具 [{name}] 执行失败: {str(e)}"}

    def remove(self, name: str):
        self._tools.pop(name, None)

    def clear(self):
        self._tools.clear()

    def __len__(self):
        return len(self._tools)
