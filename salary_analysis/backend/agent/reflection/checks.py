"""纯检测函数 — 无状态、可单独测试，输入 step 列表，输出 issue 列表"""

from __future__ import annotations

import json
import re
from typing import Any

from ..guardrails import OutputGuard


# ── 工具错误检测 ──────────────────────────────────────────────────────

def check_tool_errors(steps: list) -> list[dict]:
    """检测工具执行结果中是否包含错误。

    捕获两类错误模式：
    1. Registry 返回的结构化 {"error": "..."} 字典
    2. 并行路径 gather(return_exceptions=True) 捕获的异常字符串

    返回 list[dict]: [{check_type, severity, message, details, corrective_prompt}, ...]
    """
    ERROR_HINTS = {
        "参数校验": "请检查参数名称和取值范围是否正确",
        "未知工具": "请使用 list_tools 查看可用工具列表",
        "超时": "请简化查询条件，减少数据量",
        "连接": "请稍后重试",
        "权限": "当前没有此操作权限",
        "不存在": "请检查查询条件是否正确",
    }

    issues = []
    for step in steps:
        result = step.tool_result
        if not isinstance(result, str) or not result:
            continue

        # 尝试解析 JSON 结构化错误
        error_info = _parse_json_error(result)
        if error_info:
            detail = error_info.get("detail", str(error_info))
            hint = _match_hint(detail, ERROR_HINTS)
            issues.append({
                "check_type": "tool_error",
                "severity": "high",
                "message": f"工具 [{step.tool_name}] 执行出错: {detail[:150]}",
                "details": {
                    "tool": step.tool_name,
                    "args": step.tool_args,
                    "error_detail": detail[:500],
                    "hint": hint,
                },
                "corrective_prompt": (
                    f"## 工具执行错误\n\n"
                    f"工具 [{step.tool_name}] 执行失败:\n"
                    f"- 参数: {step.tool_args}\n"
                    f"- 错误: {detail[:200]}\n"
                    f"- 建议: {hint}\n\n"
                    f"请修正参数后重新调用，或尝试其他工具获取数据。"
                ),
            })
            continue

        # 非 JSON 错误字符串检测（如并行路径的纯文本错误）
        error_str = _detect_error_string(result)
        if error_str:
            hint = _match_hint(error_str, ERROR_HINTS)
            issues.append({
                "check_type": "tool_error",
                "severity": "high",
                "message": f"工具 [{step.tool_name}] 执行异常: {error_str[:150]}",
                "details": {
                    "tool": step.tool_name,
                    "args": step.tool_args,
                    "error_detail": error_str[:500],
                    "hint": hint,
                },
                "corrective_prompt": (
                    f"## 工具执行错误\n\n"
                    f"工具 [{step.tool_name}] 执行异常:\n"
                    f"- 错误: {error_str[:200]}\n"
                    f"- 建议: {hint}\n\n"
                    f"请尝试其他参数或工具获取数据。"
                ),
            })

    return issues


# ── 空结果检测 ────────────────────────────────────────────────────────

def check_empty_results(steps: list) -> list[dict]:
    """检测工具是否返回了空数据或异常数据。

    检测条件（任一触发）：
    - sample_count == 0
    - count == 0
    - cities/industries/results/data 列表为空
    - 统计字段全为 0
    """
    issues = []
    for step in steps:
        result = step.tool_result
        if not isinstance(result, str) or not result:
            continue

        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue

        indicators = []

        # sample_count == 0
        sc = parsed.get("sample_count")
        if sc is not None and sc == 0:
            indicators.append("sample_count=0（无样本数据）")

        # count == 0
        cnt = parsed.get("count")
        if cnt is not None and cnt == 0:
            indicators.append("count=0（无匹配记录）")

        # 空列表
        for key in ("cities", "industries", "results", "data"):
            val = parsed.get(key)
            if isinstance(val, list) and len(val) == 0:
                indicators.append(f"'{key}' 返回空列表")

        # 统计字段异常
        stats = parsed.get("statistics") or {}
        overall = stats.get("overall") if isinstance(stats, dict) else {}
        if isinstance(overall, dict):
            avg = overall.get("salary_avg", 0)
            p50 = overall.get("salary_p50", 0)
            if avg == 0 and p50 == 0 and sc != 0:
                indicators.append("统计值全为 0（数据异常）")

        if indicators:
            issues.append({
                "check_type": "empty_result",
                "severity": "medium",
                "message": f"工具 [{step.tool_name}] 返回空数据: {indicators[0]}",
                "details": {
                    "tool": step.tool_name,
                    "args": step.tool_args,
                    "indicators": indicators,
                },
                "corrective_prompt": (
                    f"## 查询结果为空\n\n"
                    f"工具 [{step.tool_name}] 返回空数据:\n"
                    f"- 参数: {step.tool_args}\n"
                    f"- 异常指标: {'; '.join(indicators)}\n\n"
                    f"请尝试:\n"
                    f"1. 使用更宽泛的查询条件（如去掉行业限制）\n"
                    f"2. 检查城市/行业名称是否正确\n"
                    f"3. 如果确实无数据，在回答中如实告知用户"
                ),
            })

    return issues


# ── 数据矛盾检测 ──────────────────────────────────────────────────────

def check_contradictions(
    steps: list,
    cache: dict[str, list[dict]] | None = None,
    threshold: float = 0.30,
) -> list[dict]:
    """检测跨工具调用的数据矛盾。

    从每个 tool_result 中提取 (industry, city, avg_salary) 等指标，
    以 (industry, city, metric_name) 为 key 缓存历史值，
    新值与历史值偏差超过 threshold 即视为矛盾。

    cache: 可选的外部历史缓存，键为 "(industry, city, metric_name)"，值为 [dict, ...]
    threshold: 偏差阈值，默认 0.30（30%）
    """
    if cache is None:
        cache = {}
    issues = []

    for step in steps:
        result = step.tool_result
        if not isinstance(result, str) or not result:
            continue

        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue

        metrics = _extract_metrics(step.tool_name, step.tool_args, parsed)
        for metric in metrics:
            key = (metric["industry"], metric["city"], "avg_salary")
            value = metric["avg_salary"]
            if value <= 0:
                continue

            if key in cache:
                for prev in cache[key]:
                    prev_val = prev.get("avg_salary", 0)
                    if prev_val <= 0:
                        continue
                    min_v = min(value, prev_val)
                    max_v = max(value, prev_val)
                    deviation = (max_v - min_v) / min_v
                    if deviation > threshold:
                        issues.append({
                            "check_type": "contradiction",
                            "severity": "medium",
                            "message": (
                                f"数据矛盾: {metric['industry']}/{metric['city']} "
                                f"平均薪资差异 {deviation * 100:.1f}% "
                                f"（{prev_val} vs {value}）"
                            ),
                            "details": {
                                "industry": metric["industry"],
                                "city": metric["city"],
                                "metric": "avg_salary",
                                "tool_a": prev.get("tool", ""),
                                "value_a": prev_val,
                                "sample_a": prev.get("sample_count", 0),
                                "tool_b": metric["tool"],
                                "value_b": value,
                                "sample_b": metric.get("sample_count", 0),
                                "deviation_pct": round(deviation * 100, 1),
                            },
                            "corrective_prompt": (
                                f"## 数据矛盾检测\n\n"
                                f"**{metric['industry']}/{metric['city']} 平均薪资** "
                                f"在不同工具返回中存在显著差异:\n"
                                f"- 工具 [{prev.get('tool', '')}]: {prev_val} 元/月 "
                                f"(样本量: {prev.get('sample_count', 0)})\n"
                                f"- 工具 [{metric['tool']}]: {value} 元/月 "
                                f"(样本量: {metric.get('sample_count', 0)})\n"
                                f"- 偏差: {deviation * 100:.1f}%\n\n"
                                f"请重新查询确认正确的数值，或在回答中向用户说明数据来源差异。"
                            ),
                        })
            # 记录到缓存
            cache.setdefault(key, []).append(metric)

    return issues


# ── 最终回答质量检测 ──────────────────────────────────────────────────

def check_output_quality(answer: str) -> list[dict]:
    """检测最终回答的质量问题。

    委托给 OutputGuard 做文本级别检查，
    额外检查是否包含具体数字和是否提及数据来源。
    """
    issues = []

    if not answer or len(answer.strip()) < 10:
        issues.append({
            "check_type": "output_quality",
            "severity": "high",
            "message": "回答为空或过短",
            "details": {},
            "corrective_prompt": (
                "## 回答质量检查\n\n"
                "回答内容为空或过短。请基于已有数据重新生成有实质内容的回答。"
            ),
        })
        return issues

    # 调用 OutputGuard
    guard_result = OutputGuard.check(answer)
    for gi in guard_result["issues"]:
        issues.append({
            "check_type": "output_quality",
            "severity": gi["severity"],
            "message": gi["message"],
            "details": gi,
            "corrective_prompt": (
                f"## 回答质量检查\n\n"
                f"检测到以下问题: {gi['message']}\n\n"
                f"请基于已有的工具调用数据重新生成回答，避免主观表述，"
                f"确保回答包含具体薪资数字并标注数据来源。"
            ),
        })

    # 额外检查：是否包含薪资数字
    numbers = re.findall(r'\d{4,}', answer)
    if not numbers:
        issues.append({
            "check_type": "output_quality",
            "severity": "medium",
            "message": "回答中未包含具体的薪资数字",
            "details": {"missing": "salary_numbers"},
            "corrective_prompt": (
                "## 回答质量检查\n\n"
                "回答中未包含任何具体的薪资数字。"
                "请基于工具返回的数据，补充具体的薪资数值和对比分析。"
            ),
        })

    return issues


# ── 内部辅助函数 ──────────────────────────────────────────────────────

def _parse_json_error(result: str) -> dict | None:
    """尝试从 JSON 结果中解析结构化错误信息"""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if "error" in parsed:
        return parsed
    return None


def _detect_error_string(result: str) -> str | None:
    """检测字符串中是否包含常见错误指示词"""
    result_lower = result.lower()
    error_markers = ["执行失败", "error", "traceback", "exception",
                     "连接失败", "timeout", "超时"]
    for marker in error_markers:
        if marker in result_lower:
            # 返回匹配附近的内容作为错误上下文
            idx = result_lower.find(marker)
            start = max(0, idx - 20)
            end = min(len(result), idx + 100)
            return result[start:end]
    return None


def _match_hint(error_detail: str, hint_map: dict) -> str:
    """根据错误细节匹配最合适的建议"""
    for keyword, hint in hint_map.items():
        if keyword in error_detail:
            return hint
    return "请检查参数后重试"


def _extract_metrics(tool_name: str, tool_args: dict, parsed: dict) -> list[dict]:
    """从工具结果中提取可用于矛盾检测的指标元组。

    返回 list[dict]，每个 dict 包含 industry, city, avg_salary, sample_count, tool
    """
    metrics = []

    # query_salary_statistics: {"industry": "...", "city": "...", "statistics": {"overall": {"salary_avg": ...}}, "sample_count": ...}
    if tool_name == "query_salary_statistics":
        industry = parsed.get("industry", tool_args.get("industry", ""))
        city = parsed.get("city", tool_args.get("city", ""))
        stats = parsed.get("statistics") or {}
        overall = stats.get("overall") if isinstance(stats, dict) else {}
        if isinstance(overall, dict):
            avg = overall.get("salary_avg", 0) or 0
            metrics.append({
                "tool": tool_name,
                "industry": industry,
                "city": city,
                "avg_salary": avg,
                "sample_count": parsed.get("sample_count", 0),
            })

    # compare_cities: {"industry": "...", "cities": [{"city": "...", "avg_salary": ..., ...}]}
    elif tool_name == "compare_cities":
        industry = parsed.get("industry", tool_args.get("industry", ""))
        cities = parsed.get("cities") or []
        if isinstance(cities, list):
            for c in cities:
                if isinstance(c, dict):
                    avg = c.get("avg_salary", 0) or 0
                    metrics.append({
                        "tool": tool_name,
                        "industry": industry,
                        "city": c.get("city", ""),
                        "avg_salary": avg,
                        "sample_count": c.get("sample_count", 0),
                    })

    # compare_industries: {"city": "...", "industries": [{"industry": "...", "avg_salary": ..., ...}]}
    elif tool_name == "compare_industries":
        city = parsed.get("city", tool_args.get("city", ""))
        industries = parsed.get("industries") or []
        if isinstance(industries, list):
            for ind in industries:
                if isinstance(ind, dict):
                    avg = ind.get("avg_salary", 0) or 0
                    metrics.append({
                        "tool": tool_name,
                        "industry": ind.get("industry", ""),
                        "city": city,
                        "avg_salary": avg,
                        "sample_count": ind.get("sample_count", 0),
                    })

    return metrics
