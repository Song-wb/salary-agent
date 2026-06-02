from .config import AIConfig
from .client import AIClient
from .context import SalaryContext
from .prompts import build_chat_prompt
from .schemas import ChatRequest, ChatResponse


class ChatEngine:
    """对话引擎：管理对话历史，调用 LLM 生成回复"""

    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()
        self.client = AIClient(self.config)
        self.context = SalaryContext()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理对话请求"""
        context_text = await self.context.build_context(request.industry, request.city)

        history = []
        if request.history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.history[-self.config.context_history * 2:]
            ]

        messages = build_chat_prompt(context_text, history, request.message)
        reply = await self.client.chat(messages)

        return ChatResponse(reply=reply)

    async def chat_stream(self, request: ChatRequest):
        """流式对话，返回异步生成器"""
        context_text = await self.context.build_context(request.industry, request.city)

        history = []
        if request.history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.history[-self.config.context_history * 2:]
            ]

        messages = build_chat_prompt(context_text, history, request.message)
        async for chunk in self.client.chat_stream(messages):
            yield chunk
