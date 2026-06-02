"""Reflector — ReAct 循环的自纠错反射器

将检测结果组装为结构化结果，生成合并的修正提示注入 messages。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .checks import (
    check_tool_errors,
    check_empty_results,
    check_contradictions,
    check_output_quality,
)


@dataclass
class ReflectorIssue:
    """单个反射检测到的问题"""
    check_type: str = ""         # "tool_error" | "empty_result" | "contradiction" | "output_quality"
    severity: str = ""           # "high" | "medium" | "low"
    message: str = ""            # 人类可读的描述
    details: dict = field(default_factory=dict)   # 结构化数据
    corrective_prompt: str = ""  # 要注入的修正提示文本


@dataclass
class ReflectorResult:
    """一轮反射检测的结果"""
    passed: bool = True
    issues: list[ReflectorIssue] = field(default_factory=list)
    corrective_messages: list[dict] = field(default_factory=list)  # [{"role": "user", "content": "..."}]


@dataclass
class ReflectorConfig:
    """反射器配置"""
    max_reflection_depth: int = 2          # 每轮 ReAct 最多反射次数
    contradiction_threshold: float = 0.30  # 数据矛盾阈值（30%）
    max_errors_to_report: int = 5          # 单轮报告中最多列出的问题数


class Reflector:
    """ReAct 循环的结构化反射器

    在工具执行后检查结果质量，在最终回答产出后验证回答质量。
    不依赖 LLM，所有检查都是确定性的规则检查。
    """

    def __init__(self, config: ReflectorConfig | None = None):
        self.config = config or ReflectorConfig()
        self._contradiction_cache: dict[str, list[dict]] = {}

    def check_tool_results(self, steps: list) -> ReflectorResult:
        """检查工具执行结果 — 在每轮工具调用后调用

        运行三类检查：
        1. 工具执行错误
        2. 空结果/异常数据
        3. 跨工具数据矛盾

        将 issue 合并为一条 corrective message 注入 messages。
        """
        issues_raw: list[dict] = []
        issues_raw.extend(check_tool_errors(steps))
        issues_raw.extend(check_empty_results(steps))
        issues_raw.extend(check_contradictions(
            steps, self._contradiction_cache, self.config.contradiction_threshold,
        ))

        return self._build_result(issues_raw)

    def check_final_answer(self, answer: str) -> ReflectorResult:
        """检查最终回答质量 — 在 ReAct 循环退出后调用

        委托给 check_output_quality 检查：
        1. 空回答
        2. 主观表述
        3. 缺少具体薪资数字
        """
        issues_raw = check_output_quality(answer)
        return self._build_result(issues_raw)

    def reset(self):
        """重置矛盾检测缓存（在每次独立 ReAct 循环前调用）"""
        self._contradiction_cache.clear()

    def _build_result(self, issues_raw: list[dict]) -> ReflectorResult:
        """将 issue dict 列表组装为 ReflectorResult"""
        if not issues_raw:
            return ReflectorResult(passed=True, issues=[], corrective_messages=[])

        # 转成 dataclass 并限制数量
        issues = []
        for ir in issues_raw[:self.config.max_errors_to_report]:
            issues.append(ReflectorIssue(
                check_type=ir["check_type"],
                severity=ir["severity"],
                message=ir["message"],
                details=ir.get("details", {}),
                corrective_prompt=ir.get("corrective_prompt", ""),
            ))

        # 合并为一条 corrective message
        combined = self._combine_prompts(issues)
        return ReflectorResult(
            passed=False,
            issues=issues,
            corrective_messages=[{"role": "user", "content": combined}],
        )

    def _combine_prompts(self, issues: list[ReflectorIssue]) -> str:
        """将多个 issue 的 corrective_prompt 合并为一条消息"""
        if len(issues) == 1:
            return issues[0].corrective_prompt

        # 按 severity 分组
        high = [i for i in issues if i.severity == "high"]
        medium = [i for i in issues if i.severity == "medium"]
        low = [i for i in issues if i.severity == "low"]

        parts = ["系统检测到以下问题，请在下一步中逐一处理：\n"]
        _append_group(parts, "工具执行错误", high)
        _append_group(parts, "数据异常或空结果", medium)
        _append_group(parts, "回答质量提醒", low)

        parts.append("\n请先修正以上错误，再继续分析。如工具连续失败，请考虑使用其他数据来源。")
        return "\n".join(parts)


def _append_group(parts: list[str], title: str, items: list[ReflectorIssue]):
    """向消息片段中添加一个 issue 分组"""
    if not items:
        return
    parts.append(f"\n== {title}（{len(items)} 个）==")
    for item in items:
        # 摘要行
        parts.append(f"- {item.message}")
        # 如果有 hint，加进去
        hint = item.details.get("hint", "")
        if hint:
            parts.append(f"  → 建议: {hint}")
