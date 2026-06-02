"""薪资数据查询工具 — 封装后端 API 为 Agent 可调用的工具

优先通过本地函数调用获取数据（避免自调用 HTTP 死锁），
回退到 HTTP 请求（Docker 跨容器场景）。

缓存策略：
  L1 (_benchmark_cache): 仅缓存本地基准路径的计算结果，永久不失效
  L2 (_process_cache):   缓存所有路径的结果（含 HTTP），TTL=600s
  L3: 由 core.py 的 ReActAgent 在调用 registry.execute 之前处理
"""

import os
import time
from fetchers.salary_benchmark import SalaryBenchmark
from agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# 本地基准数据访问（免 HTTP）
_benchmark = SalaryBenchmark()

# Docker 跨容器时走 HTTP
_INTERNAL_API = os.getenv("AI_INTERNAL_API_URL", "")

# ── 内部 HTTP 熔断器 ───────────────────────────────────────────
# Docker 模式下如果后端服务重启，熔断 HTTP 请求，降级到本地基准计算
internal_http_cb = CircuitBreaker("internal-http", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
))

# ── L1: 基准缓存 ──────────────────────────────────────────────
# key: (industry, city) → 完整 dict
# TTL: 永久（基准数据是代码常量）
_benchmark_cache: dict[tuple[str, str], dict] = {}

# ── L2: 进程缓存（带 TTL） ────────────────────────────────────
# key: (industry, city) → (cache_time, dict)
_process_cache: dict[tuple[str, str], tuple[float, dict]] = {}
PROCESS_CACHE_TTL = 600  # 10 分钟


def _use_http() -> bool:
    """是否通过 HTTP 调用（Docker 场景返回 True）"""
    return bool(_INTERNAL_API) and "backend" in _INTERNAL_API


def clear_caches():
    """清空所有缓存（测试和管理用）"""
    _benchmark_cache.clear()
    _process_cache.clear()


# ── 辅助：从缓存读/写 ─────────────────────────────────────────

def _get_from_cache(industry: str, city: str) -> dict | None:
    """二级联查（L2 → L1），命中即返回"""
    key = (industry, city)

    # L2 检查（进程级，TTL）
    now = time.time()
    if key in _process_cache:
        cached_time, cached_data = _process_cache[key]
        if now - cached_time < PROCESS_CACHE_TTL:
            return cached_data

    # L1 检查（基准缓存，仅本地路径，永久）
    if not _use_http() and key in _benchmark_cache:
        return _benchmark_cache[key]

    return None


def _write_to_caches(result: dict):
    """二级写入（L1+L2）"""
    key = (result.get("industry", ""), result.get("city", ""))

    # L1: 仅本地路径（本地路径一定从基准数据计算）
    if not _use_http():
        _benchmark_cache[key] = result

    # L2: 所有路径，带 TTL
    _process_cache[key] = (time.time(), result)


async def query_salary_statistics(industry: str = "互联网", city: str = "北京") -> dict:
    """查询指定行业和城市的薪资统计数据"""
    # ── L1+L2 联查 ──
    cached = _get_from_cache(industry, city)
    if cached is not None:
        return cached

    if _use_http():
        # ── 熔断检查：OPEN 状态跳过 HTTP，走本地基准 ──
        if not internal_http_cb.acquire():
            pass  # 跳过 HTTP，进入下面的本地基准路径
        else:
            import httpx
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{_INTERNAL_API}/api/salary/statistics",
                        params={"industry": industry, "city": city},
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    internal_http_cb.record_success()
                    _write_to_caches(result)
                    return result
            except Exception:
                internal_http_cb.record_failure()
                # 降级：走本地基准计算路径

    # 本地直接调用基准数据生成（不自调 HTTP）
    positions = _benchmark.get_positions_for_industry(industry)
    from fetchers.fetcher import _city_factor, _industry_factor, _mock_count
    cf = _city_factor(city)
    ind_f = _industry_factor(industry)
    combined = cf * ind_f

    pos_data = []
    all_avgs = []
    for pos in positions:
        sal = _benchmark.calculate_city_adjusted(pos, city, industry)
        pos_data.append({
            "position": pos,
            "avg_salary": sal["avg"],
            "count": _mock_count(pos, city, industry),
        })
        all_avgs.append(sal)

    pos_data.sort(key=lambda x: x["avg_salary"], reverse=True)

    if all_avgs:
        all_p50 = sorted(s["p50"] for s in all_avgs)
        all_avgs_sorted = sorted(s["avg"] for s in all_avgs)
        n = len(all_avgs_sorted)
        overall = {
            "salary_p25": all_p50[int(n * 0.25)] if n >= 4 else all_p50[0],
            "salary_p50": all_p50[int(n * 0.50)] if n >= 2 else all_p50[-1],
            "salary_p75": all_p50[int(n * 0.75)] if n >= 4 else all_p50[-1],
            "salary_avg": int(sum(all_avgs_sorted) / n),
        }
    else:
        overall = {"salary_p25": 0, "salary_p50": 0, "salary_p75": 0, "salary_avg": 0}

    exp_data = _benchmark.get_experience_salaries(overall["salary_avg"])
    total_count = sum(p["count"] for p in pos_data)
    source_info = _benchmark.get_data_source_description()

    result = {
        "industry": industry,
        "city": city,
        "sample_count": total_count,
        "statistics": {
            "overall": overall,
            "by_experience": exp_data,
            "by_position": pos_data,
        },
        "data_source": source_info["source_name"],
        "source_description": source_info["source_description"],
    }

    # 写入缓存
    _write_to_caches(result)
    return result


async def query_recruitment_list(city: str = "北京", limit: int = 20) -> dict:
    """查询指定城市的原始招聘数据列表"""
    if _use_http():
        # ── 熔断检查 ──
        if not internal_http_cb.acquire():
            pass  # 跳过 HTTP，走本地 mock 路径
        else:
            import httpx
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{_INTERNAL_API}/api/recruitment/list",
                        params={"city": city, "limit": limit},
                    )
                    resp.raise_for_status()
                    internal_http_cb.record_success()
                    return resp.json()
            except Exception:
                internal_http_cb.record_failure()
                # 降级：走本地 mock 数据

    # 本地直接生成样本数据
    positions = _benchmark.get_positions_for_industry("互联网")
    from fetchers.fetcher import _mock_count
    data = []
    for pos in positions[:limit]:
        sal = _benchmark.calculate_city_adjusted(pos, city, "互联网")
        data.append({
            "platform": "前程无忧",
            "position_name": pos,
            "company_name": f"{city}科技有限公司",
            "salary_min": sal["p25"],
            "salary_max": sal["p75"],
            "salary_avg": sal["avg"],
            "city": city,
            "experience": "3-5年",
            "education": "本科",
        })
    return {"count": len(data), "data": data}


SALARY_TOOL_DEFINITIONS = [
    {
        "name": "query_salary_statistics",
        "description": "查询指定行业和城市的薪资统计数据，包括整体分布、经验薪资、岗位薪资排名",
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "行业名称，可选：互联网, 金融, 制造业, 医疗健康, 教育, 房地产, 消费品, 物流",
                    "default": "互联网",
                },
                "city": {
                    "type": "string",
                    "description": "城市名称，可选：北京, 上海, 深圳, 杭州, 广州, 成都, 武汉, 南京, 西安, 重庆",
                    "default": "北京",
                },
            },
        },
    },
    {
        "name": "query_recruitment_list",
        "description": "查询指定城市的原始招聘信息列表，可限制返回条数",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称",
                    "default": "北京",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数上限",
                    "default": 20,
                },
            },
        },
    },
]
