"""熔断器（Circuit Breaker）— 三段状态机，防止级联故障

使用方式：
    cb = CircuitBreaker("my-service", CircuitBreakerConfig(failure_threshold=3))

    if not cb.acquire():
        return fallback_response()          # 熔断打开，快速降级

    try:
        result = await external_service()    # 调用外部服务
        cb.record_success()
        return result
    except Exception:
        cb.record_failure()
        raise
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class CircuitBreakerState(Enum):
    """熔断器三段状态"""
    CLOSED = "CLOSED"          # 正常 — 调用通过，统计失败次数
    OPEN = "OPEN"              # 熔断 — 请求直接拒绝
    HALF_OPEN = "HALF_OPEN"    # 半开 — 允许试探请求


@dataclass
class CircuitBreakerConfig:
    """熔断器配置

    Attributes:
        failure_threshold:   连续失败 N 次后熔断（CLOSED → OPEN）
        recovery_timeout:    熔断后等待 N 秒进入半开（OPEN → HALF_OPEN）
        half_open_max_tries: 半开状态下最多试探 N 次
        success_threshold:   连续成功 N 次后恢复（HALF_OPEN → CLOSED）
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_tries: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """三段式熔断器 — 线程安全，同步调用"""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # 状态
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_tries = 0

        # 统计
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_rejected = 0

    # ── 核心接口 ──────────────────────────────────────────────

    def acquire(self) -> bool:
        """检查是否允许这次调用通过

        Returns:
            True — 允许调用（CLOSED 或 HALF_OPEN 且有试探配额）
            False — 拒绝调用（OPEN 且未到恢复时间）
        """
        self.total_calls += 1

        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self._to_half_open()
                return True
            self.total_rejected += 1
            return False

        # HALF_OPEN
        if self.half_open_tries < self.config.half_open_max_tries:
            self.half_open_tries += 1
            return True

        self.total_rejected += 1
        return False

    def record_success(self):
        """记录一次成功调用"""
        self.success_count += 1
        self.total_successes += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self._to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # CLOSED 状态下 — 重置失败计数
            self.failure_count = 0

    def record_failure(self):
        """记录一次失败调用"""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._to_open()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._to_open()  # HALF_OPEN 失败立刻回到 OPEN

    # ── 状态查询 ──────────────────────────────────────────────

    def is_available(self) -> bool:
        """服务是否被认为可用（等效于 acquire 但不增加计数）"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                return True
            return False
        return self.half_open_tries < self.config.half_open_max_tries

    def reset(self):
        """强制重置为 CLOSED（手动恢复用）"""
        self._to_closed()

    def get_status(self) -> dict:
        """返回当前状态快照（用于监控端点）"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "half_open_tries": self.half_open_tries,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejected": self.total_rejected,
            "is_available": self.is_available(),
        }

    # ── 状态迁移 ──────────────────────────────────────────────

    def _to_open(self):
        self.state = CircuitBreakerState.OPEN
        self.failure_count = self.config.failure_threshold  # 钳位
        self.half_open_tries = 0

    def _to_half_open(self):
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_tries = 1  # 这次 acquire 本身算一次试探
        self.success_count = 0

    def _to_closed(self):
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_tries = 0
