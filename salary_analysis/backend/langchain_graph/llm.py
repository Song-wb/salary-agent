"""DeepSeek LLM — LangChain ChatOpenAI 封装

将现有的 DeepSeek API (OpenAI 兼容) 包装为 LangChain ChatModel。
集成 circuit_breaker 熔断保护，同时支持流式和非流式调用。

流式支持 (_astream):
    astream() 是 LangGraph astream_events 产生 on_chat_model_stream
    事件的前提。实现 _astream 以确保 LLM 调用通过流式 API 进行。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk

from ai.config import AIConfig
from ai.exceptions import AIUnavailableError
from agent.circuit_breaker import CircuitBreaker

logger = logging.getLogger("langchain_graph.llm")


class CircuitBreakerChatModel(BaseChatModel):
    """带熔断保护的 ChatModel 包装器

    在调用底层 LLM 之前检查 circuit_breaker 状态。
    OPEN 状态直接抛出异常，避免等待超时。

    同时支持流式 (_astream) 和非流式 (_agenerate) 调用。
    """

    llm: ChatOpenAI
    circuit_breaker: CircuitBreaker

    def __init__(self, llm: ChatOpenAI, circuit_breaker: CircuitBreaker):
        super().__init__(llm=llm, circuit_breaker=circuit_breaker)
        self.llm = llm
        self.circuit_breaker = circuit_breaker

    @property
    def _llm_type(self) -> str:
        return "deepseek_with_circuit_breaker"

    # ── 非流式调用 ──────────────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成 — 暂不支持 (使用 async 模式)"""
        raise NotImplementedError("同步调用未实现，请使用异步调用")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成 — 带熔断保护"""
        if not self.circuit_breaker.acquire():
            raise AIUnavailableError("AI 服务暂时不可用 (熔断开启)")

        try:
            result = await self.llm._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"LLM 调用失败: {type(e).__name__}: {e}")
            raise

    # ── 流式调用 (关键: LangGraph astream_events 依赖此方法产生事件) ─

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """流式生成 — 带熔断保护

        使用底层 ChatOpenAI 的流式 API，逐块 yield 生成结果。
        LangGraph 的 astream_events 会捕获每个 chunk 并产生
        on_chat_model_stream 事件，进而被 streaming.py 映射为 text 事件。
        """
        if not self.circuit_breaker.acquire():
            raise AIUnavailableError("AI 服务暂时不可用 (熔断开启)")

        try:
            async for chunk in self.llm._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"LLM 流式调用失败: {type(e).__name__}: {e}")
            raise

    @property
    def _identifying_params(self) -> dict:
        return self.llm._identifying_params


def create_llm(
    config: Optional[AIConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> CircuitBreakerChatModel:
    """创建 DeepSeek ChatModel 实例

    Args:
        config: AIConfig 实例，默认从环境变量读取
        circuit_breaker: 熔断器实例，默认使用 ai.client.deepseek_cb

    Returns:
        CircuitBreakerChatModel 实例 (支持 astream / ainvoke)
    """
    if config is None:
        config = AIConfig()

    if circuit_breaker is None:
        from ai.client import deepseek_cb
        circuit_breaker = deepseek_cb

    raw_llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.chat_temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )

    return CircuitBreakerChatModel(llm=raw_llm, circuit_breaker=circuit_breaker)
