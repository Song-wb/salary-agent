import httpx
import json
import os


class SalaryContext:
    """从后端 API 获取统计数据，组装为 LLM 可用的上下文

    内部缓存策略（实例级，辅助 L3）：
    - _cache: dict[(industry, city)] → dict
    - 存活期: 当前 SalaryContext 实例生命周期
    - 配合 app.py 的 L2 进程缓存共同减少重复 HTTP 调用
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("AI_INTERNAL_API_URL", "http://backend:8000")
        # ── L3: 实例级缓存 ──
        self._cache: dict[tuple[str, str], dict] = {}

    async def get_statistics(self, industry: str, city: str) -> dict | None:
        """调用后端统计接口获取薪资数据（带实例级缓存）"""
        key = (industry, city)
        if key in self._cache:
            return self._cache[key]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/api/salary/statistics",
                    params={"industry": industry, "city": city}
                )
                resp.raise_for_status()
                result = resp.json()
                self._cache[key] = result
                return result
        except Exception:
            return None

    async def get_samples(self, city: str, limit: int = 10) -> list:
        """获取原始招聘样本数据"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/api/recruitment/list",
                    params={"city": city, "limit": limit}
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("data", [])
        except Exception:
            return []

    async def build_context(self, industry: str, city: str) -> str:
        """组装统计数据和样本数据为上下文文本"""
        stats = await self.get_statistics(industry, city)
        if not stats:
            return ""

        lines = []
        statistics = stats.get("statistics", {})
        overall = statistics.get("overall", {})
        sample_count = stats.get("sample_count", 0)
        data_source = stats.get("data_source", "未知")
        lines.append(f"行业：{stats.get('industry', industry)}")
        lines.append(f"城市：{stats.get('city', city)}")
        lines.append(f"样本量：{sample_count} 条")
        lines.append(f"数据来源：{data_source}")

        if overall:
            lines.append(f"25分位薪资：{overall.get('salary_p25', 0)} 元/月")
            lines.append(f"中位数薪资：{overall.get('salary_p50', 0)} 元/月")
            lines.append(f"75分位薪资：{overall.get('salary_p75', 0)} 元/月")
            lines.append(f"平均薪资：{overall.get('salary_avg', 0)} 元/月")

        by_exp = statistics.get("by_experience", [])
        if by_exp:
            lines.append("\n经验-薪资分布：")
            for item in by_exp:
                lines.append(f"  - {item.get('experience')}：平均 {item.get('avg_salary')} 元/月（{item.get('count')} 个样本）")

        by_pos = statistics.get("by_position", [])
        if by_pos:
            lines.append("\n岗位薪资排名：")
            for item in by_pos:
                lines.append(f"  - {item.get('position')}：平均 {item.get('avg_salary')} 元/月（{item.get('count')} 个样本）")

        return "\n".join(lines)
