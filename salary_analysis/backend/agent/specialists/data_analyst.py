"""数据分析师 Agent — 专注于数据查询、统计分析和对比"""

from ..core import ReActAgent, AgentResult, DEFAULT_SYSTEM_PROMPT

DATA_ANALYST_PROMPT = DEFAULT_SYSTEM_PROMPT + """

你是**数据分析师**，专精于薪资数据的查询和统计分析。

你的工作方式：
1. 理解分析需求，确定需要哪些数据
2. 使用工具获取原始数据
3. 对数据进行对比、排序、趋势分析
4. 总结发现，给出数据驱动的结论

你可以使用的工具：
- query_salary_statistics: 查询单行业/城市的薪资统计
- query_recruitment_list: 查询原始招聘信息
- compare_cities: 跨城市对比
- compare_industries: 跨行业对比

分析时请注意：
- 对比时要控制变量（如对比不同城市时保持行业一致）
- 关注样本量，样本过少时给出提醒
- 不仅报告数据，还要解读数据背后的含义
"""


class DataAnalystAgent:
    """数据分析师 Agent"""

    def __init__(self, config, registry):
        self.agent = ReActAgent(config, registry)
        self.system_prompt = DATA_ANALYST_PROMPT

    async def analyze(self, question: str, context_text: str = "",
                       session_cache: dict | None = None) -> AgentResult:
        """执行数据分析任务，返回完整 AgentResult"""
        result = await self.agent.run(
            user_message=question,
            system_prompt=self.system_prompt,
            context_text=context_text,
            max_steps=8,
            session_cache=session_cache,
        )
        return result
