"""分析工具 — 多维度对比分析"""

import asyncio
import os

INTERNAL_API = os.getenv("AI_INTERNAL_API_URL", "http://backend:8000")

# 支持的城市和行业（用于验证参数）
SUPPORTED_CITIES = ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "南京", "西安", "重庆"]
SUPPORTED_INDUSTRIES = ["互联网", "金融", "制造业", "医疗健康", "教育", "房地产", "消费品", "物流"]


async def _fetch_stats(industry: str, city: str) -> dict | None:
    """获取单条统计数据（优先本地直接计算）"""
    try:
        # Docker 跨容器时走 HTTP
        if "backend" in INTERNAL_API:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{INTERNAL_API}/api/salary/statistics",
                    params={"industry": industry, "city": city},
                )
                resp.raise_for_status()
                return resp.json()

        # 本地直接调用基准数据
        from .salary import query_salary_statistics
        return await query_salary_statistics(industry=industry, city=city)
    except Exception:
        return None


async def compare_cities(industry: str = "互联网", cities: list[str] | None = None) -> dict:
    """对比多个城市的同行业薪资数据（并行查询）"""
    target_cities = [c for c in (cities or ["北京", "上海", "深圳", "杭州", "广州"])
                     if c in SUPPORTED_CITIES]

    # 并行发起所有城市的查询
    all_data = await asyncio.gather(
        *[_fetch_stats(industry, city) for city in target_cities],
        return_exceptions=True,
    )

    results = []
    for city, data in zip(target_cities, all_data):
        if isinstance(data, Exception):
            continue  # 单个城市失败不影响其他
        if data and "statistics" in data:
            overall = data["statistics"].get("overall", {})
            results.append({
                "city": city,
                "avg_salary": overall.get("salary_avg", 0),
                "median_salary": overall.get("salary_p50", 0),
                "sample_count": data.get("sample_count", 0),
            })
    results.sort(key=lambda x: x["avg_salary"], reverse=True)
    return {
        "industry": industry,
        "cities": results,
        "comparison": "各城市薪资对比（从高到低排序）",
    }


async def compare_industries(city: str = "北京", industries: list[str] | None = None) -> dict:
    """对比同一城市的多行业薪资数据（并行查询）"""
    target_industries = [i for i in (industries or ["互联网", "金融", "制造业", "医疗健康", "教育"])
                         if i in SUPPORTED_INDUSTRIES]

    # 并行发起所有行业的查询
    all_data = await asyncio.gather(
        *[_fetch_stats(ind, city) for ind in target_industries],
        return_exceptions=True,
    )

    results = []
    for ind, data in zip(target_industries, all_data):
        if isinstance(data, Exception):
            continue
        if data and "statistics" in data:
            overall = data["statistics"].get("overall", {})
            results.append({
                "industry": ind,
                "avg_salary": overall.get("salary_avg", 0),
                "median_salary": overall.get("salary_p50", 0),
                "sample_count": data.get("sample_count", 0),
            })
    results.sort(key=lambda x: x["avg_salary"], reverse=True)
    return {
        "city": city,
        "industries": results,
        "comparison": "各行业薪资对比（从高到低排序）",
    }


ANALYSIS_TOOL_DEFINITIONS = [
    {
        "name": "compare_cities",
        "description": "对比同一行业在不同城市的薪资水平，返回各城市平均薪资和中位数，按薪资从高到低排序",
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "行业名称",
                    "default": "互联网",
                },
                "cities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要对比的城市列表，不传则默认对比主要城市",
                    "default": ["北京", "上海", "深圳", "杭州", "广州"],
                },
            },
        },
    },
    {
        "name": "compare_industries",
        "description": "对比同一城市不同行业的薪资水平，返回各行业平均薪资和中位数，按薪资从高到低排序",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称",
                    "default": "北京",
                },
                "industries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要对比的行业列表，不传则默认对比主要行业",
                    "default": ["互联网", "金融", "制造业", "医疗健康", "教育"],
                },
            },
        },
    },
]
