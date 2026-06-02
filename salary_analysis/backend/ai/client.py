"""OpenAI 兼容客户端封装 — DeepSeek API

熔断保护：deepseek_cb 防止 API 不可用时级联超时。
连续 3 次失败 → 熔断 30s → 快速降级。
"""

import asyncio
from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError, APITimeoutError

from .config import AIConfig
from .exceptions import (
    APIAuthError, APIRateLimitError, APITimeoutError,
    AIError, AIUnavailableError,
)
from agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# ── DeepSeek API 熔断器 ──────────────────────────────────────
# threshold=3: 连续 3 次失败即熔断
# timeout=30s: 30 秒后尝试恢复
deepseek_cb = CircuitBreaker("deepseek-api", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_max_tries=1,
    success_threshold=2,
))


class AIClient:
    """封装 OpenAI 兼容 API 客户端 (DeepSeek)"""

    def __init__(self, config: AIConfig):
        self.config = config
        if config.is_configured:
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        else:
            self.client = None

    async def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """非流式对话（带 DeepSeek API 熔断）"""
        if not self.client:
            return "AI 服务未配置，请在 .env 中设置 DEEPSEEK_API_KEY"

        # ── 熔断检查：OPEN 状态直接快速失败 ──
        if not deepseek_cb.acquire():
            raise AIUnavailableError("AI 服务暂时不可用，请稍后再试")

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.chat_temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
            )
            deepseek_cb.record_success()
            return response.choices[0].message.content or ""

        except AuthenticationError:
            deepseek_cb.record_failure()
            raise APIAuthError("DeepSeek API 认证失败，请检查 API Key 配置")
        except RateLimitError:
            deepseek_cb.record_failure()
            raise APIRateLimitError("请求过于频繁，请稍后再试")
        except APITimeoutError:
            deepseek_cb.record_failure()
            raise APITimeoutError("AI 响应超时，请简化问题或稍后再试")
        except APIError as e:
            deepseek_cb.record_failure()
            raise AIError(f"AI API 调用失败: {e}")

    async def chat_stream(self, messages: list[dict], temperature: float | None = None):
        """流式对话（带 DeepSeek API 熔断），返回异步生成器"""
        if not self.client:
            yield "AI 服务未配置，请在 .env 中设置 DEEPSEEK_API_KEY"
            return

        # ── 熔断检查：OPEN 状态直接降级 ──
        if not deepseek_cb.acquire():
            yield "AI 服务暂时不可用，请稍后再试"
            return

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.chat_temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            deepseek_cb.record_success()

        except AuthenticationError:
            deepseek_cb.record_failure()
            yield "DeepSeek API 认证失败，请检查 API Key 配置"
        except RateLimitError:
            deepseek_cb.record_failure()
            yield "请求过于频繁，请稍后再试"
        except APITimeoutError:
            deepseek_cb.record_failure()
            yield "AI 响应超时，请简化问题或稍后再试"
        except APIError as e:
            deepseek_cb.record_failure()
            yield f"AI API 调用失败: {e}"
