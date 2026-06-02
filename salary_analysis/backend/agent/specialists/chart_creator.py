"""图表创建师 Agent — 专注于数据可视化策略和图表描述"""

from ..core import ReActAgent, AgentResult, DEFAULT_SYSTEM_PROMPT

CHART_CREATOR_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**图表创建师**，专精于确定最佳的数据可视化方案。

你的工作方式：
1. 理解数据和要传达的信息
2. 选择合适的图表类型
3. 描述图表的结构、数据映射和样式

图表类型选择指南：
- 薪资对比 → 柱状图（bar chart）
- 薪资分布 → 箱线图（box plot）或直方图
- 趋势变化 → 折线图（line chart）
- 占比关系 → 饼图（pie chart）或环形图
- 多维度 → 分组柱状图或热力图

输出格式：描述你推荐的图表方案，包括：
- 图表类型
- X轴/Y轴映射
- 数据来源
- 期望传达的洞察
"""


class ChartCreatorAgent:
    """图表创建师 Agent"""

    def __init__(self, config, registry):
        self.agent = ReActAgent(config, registry)
        self.system_prompt = CHART_CREATOR_PROMPT

    async def design(self, data_description: str, purpose: str = "",
                     context_text: str = "",
                     session_cache: dict | None = None) -> AgentResult:
        """设计图表方案，返回完整 AgentResult"""
        message = f"数据描述：{data_description}\n可视化目的：{purpose}\n请设计合适的图表方案"
        result = await self.agent.run(
            user_message=message,
            system_prompt=self.system_prompt,
            context_text=context_text,
            max_steps=5,
            session_cache=session_cache,
        )
        return result
