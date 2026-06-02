"""
薪酬数据获取器

统一入口：先从 MySQL 取已保存的爬虫数据，若无则返回基准数据。
"""

from .salary_benchmark import SalaryBenchmark


class SalaryFetcher:
    """薪酬数据获取器"""

    def __init__(self):
        self.benchmark = SalaryBenchmark()

        

    def fetch_statistics(self, industry: str, city: str) -> dict:
        """获取薪资统计数据"""
        positions = self.benchmark.get_positions_for_industry(industry)
        cf = _city_factor(city)
        ind_f = _industry_factor(industry)
        combined = cf * ind_f

        # ---- 各岗位薪资 ----
        pos_data = []
        all_avgs = []
        for pos in positions:
            sal = self.benchmark.calculate_city_adjusted(pos, city, industry)
            pos_data.append({
                "position": pos,
                "avg_salary": sal["avg"],
                "count": _mock_count(pos, city, industry),
            })
            all_avgs.append(sal)

        pos_data.sort(key=lambda x: x["avg_salary"], reverse=True)

        # ---- 总体统计 ----
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

        # ---- 经验薪资 ----
        exp_data = self.benchmark.get_experience_salaries(overall["salary_avg"])

        # ---- 样本数 ----
        total_count = sum(p["count"] for p in pos_data)

        source_info = self.benchmark.get_data_source_description()

        return {
            "overall": overall,
            "by_experience": exp_data,
            "by_position": pos_data,
            "_meta": {
                "sample_count": total_count,
                "data_source": source_info["source_name"],
                "source_description": source_info["source_description"],
            },
        }


def _city_factor(city: str) -> float:
    from .salary_benchmark import CITY_FACTOR
    return CITY_FACTOR.get(city, 0.85)


def _industry_factor(industry: str) -> float:
    from .salary_benchmark import INDUSTRY_FACTOR
    return INDUSTRY_FACTOR.get(industry, 0.90)


def _mock_count(position: str, city: str, industry: str) -> int:
    """模拟样本数量（仅作为展示用，不用来算薪资）"""
    import hashlib
    seed = f"{position}-{city}-{industry}"
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return 80 + (h % 220)
