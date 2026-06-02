from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    history: list[ChatMessage] = Field(default_factory=list, description="对话历史")
    industry: str = Field(default="互联网", description="行业")
    city: str = Field(default="北京", description="城市")


class ChatResponse(BaseModel):
    reply: str
    generated_at: datetime = Field(default_factory=datetime.now)


class ReportRequest(BaseModel):
    industry: str = Field(default="互联网", description="行业")
    city: str = Field(default="北京", description="城市")


class ReportResponse(BaseModel):
    report: str
    industry: str
    city: str
    sample_count: int
    generated_at: datetime = Field(default_factory=datetime.now)
