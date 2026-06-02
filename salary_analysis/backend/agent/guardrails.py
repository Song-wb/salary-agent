"""安全护栏 — 输入验证、输出过滤、操作确认"""

import re
from typing import Any

# Prompt injection 关键词
INJECTION_PATTERNS = [
    r"忽略(上面|之前|系统).*指令",
    r"ignore.*(above|previous|system).*instruction",
    r"你(现在|接下来).*(是|扮演)",
    r"你是.*(而不是|不是).*助手",
    r"forget.*(instruction|prompt)",
    r"system.*prompt.*override",
    r"你不需要.*(遵守|遵循|按照)",
]

# 敏感信息模式
SENSITIVE_PATTERNS = [
    r"\b\d{18}[\dXx]\b",           # 身份证号
    r"\b1[3-9]\d{9}\b",            # 手机号
    r"\b\d{6}\b",                   # 简单密码/验证码（6位数字，谨慎使用）
]


class InputGuard:
    """输入护栏 — 检测 prompt injection 和敏感信息"""

    @staticmethod
    def check(text: str) -> dict:
        """检查输入，返回检查结果"""
        issues = []

        # 检测注入
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                issues.append({
                    "type": "injection",
                    "severity": "high",
                    "message": "检测到可能的 Prompt Injection",
                    "matched": pattern,
                })
                break

        # 检测敏感信息
        for pattern in SENSITIVE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                issues.append({
                    "type": "sensitive_info",
                    "severity": "medium",
                    "message": f"检测到可能的敏感信息 ({len(matches)} 处)",
                })
                break

        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }


class OutputGuard:
    """输出护栏 — 检查输出质量"""

    # 缺乏数据支撑的断言关键词
    UNSUPPORTED_CLAIMS = [
        "根据我的经验",
        "据我所知",
        "我认为",
        "可能",
        "也许",
        "大概",
    ]

    @staticmethod
    def check(text: str, data_available: bool = True) -> dict:
        """检查输出，返回检查结果"""
        issues = []

        # 检查空输出
        if not text or len(text.strip()) < 10:
            issues.append({
                "type": "empty_output",
                "severity": "high",
                "message": "输出过短或为空",
            })

        # 检查缺乏数据支撑的断言
        for phrase in OutputGuard.UNSUPPORTED_CLAIMS:
            if phrase in text:
                issues.append({
                    "type": "unsupported_claim",
                    "severity": "low",
                    "message": f"包含主观表述: '{phrase}'",
                    "matched": phrase,
                })
                break

        return {
            "passed": len([i for i in issues if i["severity"] != "low"]) == 0,
            "issues": issues,
        }


def sanitize_input(text: str, max_length: int = 4000) -> str:
    """清洗输入：截断 + 移除控制字符"""
    # 移除控制字符（保留常见格式化字符）
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned[:max_length]
