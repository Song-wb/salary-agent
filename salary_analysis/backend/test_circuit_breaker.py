"""
CircuitBreaker 单元测试 — 三段状态机验证
"""

import time
from agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState

PASS = "[OK]"
FAIL = "[FAIL]"


def ok(msg):
    print(f"  {PASS} {msg}")


def fail(msg):
    print(f"  {FAIL} {msg}")


def test_basic():
    """CLOSED 状态：acquire 允许通过，record_success 重置失败计数"""
    print("\n=== TEST: CB Basic ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=10,
    ))

    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.acquire() is True
    ok("CLOSED: acquire() returns True")

    cb.record_success()
    assert cb.failure_count == 0
    ok("CLOSED: record_success() resets failure_count")

    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    assert cb.state == CircuitBreakerState.CLOSED
    ok("CLOSED: 2 failures still CLOSED (threshold=3)")

    cb.record_success()
    assert cb.failure_count == 0
    ok("CLOSED: record_success() resets failure count to 0 at threshold=2")


def test_open_after_threshold():
    """连续失败达阈值 → OPEN"""
    print("\n=== TEST: CB CLOSED → OPEN ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=10,
    ))

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.failure_count >= 3
    ok(f"CLOSED→OPEN: state={cb.state.value} after {cb.failure_count} failures")

    assert cb.acquire() is False
    ok("OPEN: acquire() returns False")


def test_open_recovery_timeout():
    """OPEN → 超时 → HALF_OPEN → acquire 允许"""
    print("\n=== TEST: CB OPEN → HALF_OPEN ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=0.05,  # 50ms
    ))

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.acquire() is False

    # 等待超时
    time.sleep(0.06)
    assert cb.acquire() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN
    ok("OPEN→HALF_OPEN: acquire() returns True after recovery_timeout")


def test_half_open_success_to_closed():
    """HALF_OPEN → 连续成功 → CLOSED"""
    print("\n=== TEST: CB HALF_OPEN → CLOSED ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=0.05,
        success_threshold=2,
    ))

    # 触发熔断
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    # 等待恢复
    time.sleep(0.06)
    assert cb.acquire() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN
    ok("  HALF_OPEN entered after recovery")

    # 连续成功
    cb.record_success()
    assert cb.state == CircuitBreakerState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    ok("HALF_OPEN→CLOSED: 2 successes transition to CLOSED")


def test_half_open_failure_to_open():
    """HALF_OPEN → 试探失败 → OPEN"""
    print("\n=== TEST: CB HALF_OPEN → OPEN ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=0.05,
    ))

    # 触发熔断
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.06)
    assert cb.acquire() is True
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 试探失败
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN
    assert cb.acquire() is False
    ok("HALF_OPEN→OPEN: failure returns to OPEN, acquire() returns False")


def test_reset():
    """reset() 强制回到 CLOSED"""
    print("\n=== TEST: CB Reset ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=10,
    ))

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    cb.reset()
    assert cb.state == CircuitBreakerState.CLOSED
    assert cb.failure_count == 0
    assert cb.acquire() is True
    ok("reset(): force back to CLOSED, acquire() returns True")


def test_get_status():
    """get_status() 返回正确结构"""
    print("\n=== TEST: CB get_status ===")
    cb = CircuitBreaker("test-api", CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=30,
    ))

    status = cb.get_status()
    assert status["name"] == "test-api"
    assert status["state"] == "CLOSED"
    assert status["failure_count"] == 0
    assert "total_calls" in status
    assert "total_failures" in status
    assert "total_rejected" in status
    assert "is_available" in status
    assert status["is_available"] is True

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    status = cb.get_status()
    assert status["state"] == "OPEN"
    assert status["is_available"] is False
    assert status["total_failures"] == 3
    ok("get_status(): returns correct state and stats")


def test_half_open_max_tries():
    """HALF_OPEN 多试探次数"""
    print("\n=== TEST: CB Half-Open Max Tries ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=0.05,
        half_open_max_tries=3, success_threshold=1,
    ))

    # 触发熔断
    cb.record_failure()
    cb.record_failure()
    time.sleep(0.06)

    # HALF_OPEN 允许最多 3 次试探（过渡算第 1 次）
    assert cb.acquire() is True   # 试探 1/3（OPEN→HALF_OPEN 过渡）
    assert cb.acquire() is True   # 试探 2/3
    assert cb.acquire() is True   # 试探 3/3
    assert cb.acquire() is False  # 第 4 次拒绝
    ok("HALF_OPEN: respects half_open_max_tries=3")

    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED
    ok("HALF_OPEN: success within tries recovers to CLOSED")


def test_is_available():
    """is_available() 不改变计数"""
    print("\n=== TEST: CB is_available ===")
    cb = CircuitBreaker("test", CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=0.05,
    ))

    assert cb.is_available() is True
    assert cb.total_calls == 0  # is_available 不增加计数
    ok("CLOSED: is_available() returns True, does not increment total_calls")

    cb.record_failure()
    cb.record_failure()
    assert cb.is_available() is False
    assert cb.total_calls == 0
    ok("OPEN: is_available() returns False")

    time.sleep(0.06)
    assert cb.is_available() is True  # HALF_OPEN 自动恢复
    ok("HALF_OPEN: is_available() returns True after timeout")


def test_default_config():
    """默认配置合理"""
    print("\n=== TEST: CB Default Config ===")
    cb = CircuitBreaker("default")
    assert cb.config.failure_threshold == 5
    assert cb.config.recovery_timeout == 30.0
    assert cb.config.half_open_max_tries == 3
    assert cb.config.success_threshold == 2
    ok("default config: reasonable values")


if __name__ == "__main__":
    tests = [
        test_basic,
        test_open_after_threshold,
        test_open_recovery_timeout,
        test_half_open_success_to_closed,
        test_half_open_failure_to_open,
        test_reset,
        test_get_status,
        test_half_open_max_tries,
        test_is_available,
        test_default_config,
    ]

    print("=" * 50)
    print("CircuitBreaker Test Suite")
    print("=" * 50)
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  {FAIL} {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    total = len(tests)
    passed = total - failed
    print(f"Result: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    exit(1 if failed > 0 else 0)
