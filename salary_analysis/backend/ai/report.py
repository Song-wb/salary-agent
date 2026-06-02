from .config import AIConfig
from .client import AIClient
from .context import SalaryContext
from .prompts import build_report_prompt
from .schemas import ReportRequest, ReportResponse


class ReportGenerator:
    """报告生成引擎：获取统计数据，调用 LLM 生成结构化工报告"""

    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()
        self.client = AIClient(self.config)
        self.context = SalaryContext()
        self._cache: dict[str, tuple[str, str]] = {}  # key: (industry,city) -> report

    async def generate(self, request: ReportRequest) -> ReportResponse:
        """生成薪资分析报告"""
        cache_key = f"{request.industry}:{request.city}"
        cached = self._cache.get(cache_key)
        if cached:
            report_text, source = cached
            sample_count = 0
            stats = await self.context.get_statistics(request.industry, request.city)
            if stats:
                sample_count = stats.get("sample_count", 0)
            return ReportResponse(
                report=report_text,
                industry=request.industry,
                city=request.city,
                sample_count=sample_count,
            )

        stats = await self.context.get_statistics(request.industry, request.city)
        sample_count = stats.get("sample_count", 0) if stats else 0

        context_text = await self.context.build_context(request.industry, request.city)
        messages = build_report_prompt(context_text, request.industry, request.city)

        report_text = await self.client.chat(
            messages,
            temperature=self.config.report_temperature
        )

        self._cache[cache_key] = (report_text, request.industry)
        return ReportResponse(
            report=report_text,
            industry=request.industry,
            city=request.city,
            sample_count=sample_count,
        )

    def clear_cache(self):
        """清除报告缓存"""
        self._cache.clear()
