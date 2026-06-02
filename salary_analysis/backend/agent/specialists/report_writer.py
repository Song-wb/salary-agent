"""报告撰写师 Agent — 专注于生成结构化分析报告"""

from ..core import ReActAgent, AgentResult, DEFAULT_SYSTEM_PROMPT

REPORT_WRITER_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**报告撰写师**，专精于将数据组织为结构化的专业分析报告。

你的工作方式：
1. 先获取或确认数据已就绪
2. 设计报告结构（概述、分析、结论）
3. 撰写完整的 Markdown 报告

报告要求：
- 使用 Markdown 格式，包含标题层级
- 每个章节必须有数据支撑
- 包含数据解读和洞察
- 最后给出结论和建议

报告标准结构：
## 1. 市场概况
## 2. 核心数据发现
## 3. 详细分析
## 4. 趋势与洞察
## 5. 结论与建议

如果数据不足，请先使用工具获取数据再撰写报告。
"""


class ReportWriterAgent:
    """报告撰写师 Agent"""

    def __init__(self, config, registry):
        self.agent = ReActAgent(config, registry)
        self.system_prompt = REPORT_WRITER_PROMPT

    async def write(self, topic: str, context_text: str = "",
                     session_cache: dict | None = None) -> AgentResult:
        """生成分析报告，返回完整 AgentResult"""
        result = await self.agent.run(
            user_message=topic,
            system_prompt=self.system_prompt,
            context_text=context_text,
            max_steps=8,
            session_cache=session_cache,
        )
        return result
