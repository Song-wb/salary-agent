"""可观测性 — 结构化日志、追踪、指标收集"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

# 配置 Agent 专用日志
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)


class TraceContext:
    """请求追踪上下文"""

    def __init__(self):
        self.trace_id: str = ""
        self.start_time: float = 0.0
        self.steps: list[dict] = []
        self.token_usage: dict = {}
        self.metadata: dict = {}

    def start(self, metadata: dict | None = None):
        """开始新的追踪"""
        self.trace_id = uuid.uuid4().hex[:12]
        self.start_time = time.time()
        self.steps.clear()
        self.token_usage = {}
        self.metadata = metadata or {}

    def log_step(self, step_type: str, data: dict):
        """记录一个步骤"""
        step = {
            "type": step_type,
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(time.time() - self.start_time, 3),
            **data,
        }
        self.steps.append(step)
        _write_log("STEP", self.trace_id, step)

    def log_tool_call(self, tool_name: str, args: dict, result: Any, duration: float):
        """记录工具调用"""
        self.log_step("tool_call", {
            "tool": tool_name,
            "args": _truncate(json.dumps(args, ensure_ascii=False), 500),
            "result_summary": _truncate(str(result), 200),
            "duration_ms": round(duration * 1000),
        })

    def log_llm_call(self, model: str, prompt_tokens: int,
                     completion_tokens: int, duration: float):
        """记录 LLM 调用"""
        self.token_usage["prompt_tokens"] = \
            self.token_usage.get("prompt_tokens", 0) + prompt_tokens
        self.token_usage["completion_tokens"] = \
            self.token_usage.get("completion_tokens", 0) + completion_tokens
        self.token_usage["total_tokens"] = \
            self.token_usage.get("total_tokens", 0) + prompt_tokens + completion_tokens

        self.log_step("llm_call", {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": round(duration * 1000),
        })

    def finish(self, final_answer: str = "") -> dict:
        """完成追踪，返回汇总报告"""
        total_time = round(time.time() - self.start_time, 3)
        report = {
            "trace_id": self.trace_id,
            "total_time_s": total_time,
            "steps": len(self.steps),
            "tool_calls": sum(1 for s in self.steps if s["type"] == "tool_call"),
            "llm_calls": sum(1 for s in self.steps if s["type"] == "llm_call"),
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }
        _write_log("DONE", self.trace_id, report)
        return report


class AgentMetrics:
    """全局 Agent 指标收集"""

    def __init__(self):
        self.total_sessions = 0
        self.total_tool_calls = 0
        self.total_llm_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.sessions: list[dict] = []

    def record(self, report: dict):
        """记录一次完成的追踪报告"""
        self.total_sessions += 1
        self.total_tool_calls += report.get("tool_calls", 0)
        self.total_llm_calls += report.get("llm_calls", 0)
        self.total_tokens += report.get("token_usage", {}).get("total_tokens", 0)
        self.total_time += report.get("total_time_s", 0)
        self.sessions.append(report)

    def summary(self) -> dict:
        """返回指标摘要"""
        n = max(self.total_sessions, 1)
        return {
            "total_sessions": self.total_sessions,
            "total_tool_calls": self.total_tool_calls,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time, 1),
            "avg_tokens_per_session": round(self.total_tokens / n),
            "avg_time_per_session_s": round(self.total_time / n, 2),
            "avg_tool_calls_per_session": round(self.total_tool_calls / n, 1),
        }


# 全局指标实例
global_metrics = AgentMetrics()


def _write_log(level: str, trace_id: str, data: dict):
    """写入结构化日志"""
    log_entry = {
        "level": level,
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(),
        **data,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text
