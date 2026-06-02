import os
from dataclasses import dataclass, field


@dataclass
class AIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "sk-30a62ac6de364c65a460cfe199190bba"))
    base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    chat_temperature: float = field(default_factory=lambda: float(os.getenv("AI_CHAT_TEMPERATURE", "0.7")))
    report_temperature: float = field(default_factory=lambda: float(os.getenv("AI_REPORT_TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("AI_MAX_TOKENS", "4096")))
    timeout: int = field(default_factory=lambda: int(os.getenv("AI_REQUEST_TIMEOUT", "60")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("AI_MAX_RETRIES", "3")))
    context_history: int = field(default_factory=lambda: int(os.getenv("AI_CONTEXT_HISTORY", "3")))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "sk-30a62ac6de364c65a460cfe199190bba"
