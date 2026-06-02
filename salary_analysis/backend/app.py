from fastapi import FastAPI, Request
import os
import json
import time
import pymysql
from dotenv import load_dotenv
from fetchers import SalaryFetcher
from agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

load_dotenv()  # 加载 .env 文件中的环境变量

# ── L2: API 端点进程级缓存 ──────────────────────────────────────
# key: (industry, city) → (cache_time, dict)
# TTL: 600s，collect.py 爬取期间数据可能变化
_stats_cache: dict[tuple[str, str], tuple[float, dict]] = {}
STATS_CACHE_TTL = 600

# ── MySQL 熔断器 ───────────────────────────────────────────────
# threshold=2: 连续 2 次连接失败即熔断
# timeout=60s: 1 分钟后尝试恢复
mysql_cb = CircuitBreaker("mysql", CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=60.0,
))

# ===== Agent 核心 =====
from agent.tools import build_agent_tools
from agent.orchestrator import Orchestrator
from agent.memory import WorkingMemory
from agent.observability import TraceContext, global_metrics
from agent.guardrails import InputGuard, OutputGuard, sanitize_input
from ai.config import AIConfig

# 初始化 Agent（单例）
_agent_config = AIConfig()
_agent_tools = build_agent_tools()
orchestrator = Orchestrator(_agent_config, _agent_tools) if _agent_config.is_configured else None

_salary_fetcher = SalaryFetcher()

app = FastAPI(title="薪资分析API", version="1.0")

def get_db_connection():
    """获取 MySQL 连接（带熔断保护）

    熔断 OPEN 时：立即抛 ConnectionError → 调用方自行 fallback 到基准数据。
    避免了 TCP 连接超时的等待（通常 10-30s）。
    """
    if not mysql_cb.acquire():
        raise ConnectionError("MySQL 服务暂时不可用（熔断降级）")

    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "db"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "@Swb112988"),
            database=os.getenv("DB_NAME", "salary_analysis"),
            charset="utf8mb4",
            connect_timeout=5,
        )
        mysql_cb.record_success()
        return conn
    except pymysql.Error:
        mysql_cb.record_failure()
        raise


@app.get("/init-db")
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recruitment_info (
            id INT AUTO_INCREMENT PRIMARY KEY,
            platform VARCHAR(50),
            position_name VARCHAR(100),
            company_name VARCHAR(100),
            salary_min INT,
            salary_max INT,
            salary_avg DECIMAL(10,2),
            city VARCHAR(50),
            experience VARCHAR(20),
            education VARCHAR(20),
            publish_time VARCHAR(20),
            crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salary_statistics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            industry VARCHAR(50),
            position VARCHAR(100),
            city VARCHAR(50),
            experience VARCHAR(20),
            education VARCHAR(20),
            sample_count INT,
            salary_p25 INT,
            salary_p50 INT,
            salary_p75 INT,
            salary_avg DECIMAL(10,2),
            stat_time DATE DEFAULT CURRENT_DATE
        )
    """)
    conn.commit()
    conn.close()
    return {"message": "数据库初始化成功"}


@app.get("/api/salary/statistics")
def get_salary_statistics(industry: str = "互联网", city: str = "北京"):
    """获取薪资统计数据（带 L2 进程级缓存，TTL=600s）"""
    key = (industry, city)
    now = time.time()

    # ── L2 检查 ──
    if key in _stats_cache:
        cached_time, cached_data = _stats_cache[key]
        if now - cached_time < STATS_CACHE_TTL:
            return cached_data

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        cursor.execute("""
            SELECT * FROM recruitment_info
            WHERE city = %s
            ORDER BY crawl_time DESC
            LIMIT 1000
        """, (city,))

        rows = cursor.fetchall()

        if rows:
            salaries = [row['salary_avg'] for row in rows if row['salary_avg']]
            by_experience = {}
            by_position = {}

            for row in rows:
                exp = row['experience'] or '未知'
                pos = row['position_name'][:20] if row['position_name'] else '未知'
                by_experience.setdefault(exp, []).append(row['salary_avg'])
                by_position.setdefault(pos, []).append(row['salary_avg'])

            salaries.sort()
            n = len(salaries)
            p25 = salaries[int(n * 0.25)] if n > 0 else 0
            p50 = salaries[int(n * 0.5)] if n > 0 else 0
            p75 = salaries[int(n * 0.75)] if n > 0 else 0
            avg = sum(salaries) / n if n > 0 else 0

            exp_data = [
                {"experience": exp, "avg_salary": int(sum(vals) / len(vals)), "count": len(vals)}
                for exp, vals in by_experience.items() if len(vals) >= 3
            ]
            exp_data.sort(key=lambda x: ['0-1年', '1-3年', '3-5年', '5年+'].index(x['experience'])
                          if x['experience'] in ['0-1年', '1-3年', '3-5年', '5年+'] else 99)

            pos_data = [
                {"position": pos, "avg_salary": int(sum(vals) / len(vals)), "count": len(vals)}
                for pos, vals in by_position.items() if len(vals) >= 3
            ]
            pos_data.sort(key=lambda x: x['avg_salary'], reverse=True)
            pos_data = pos_data[:12]

            result = {
                "industry": industry,
                "city": city,
                "sample_count": n,
                "statistics": {
                    "overall": {"salary_p25": int(p25), "salary_p50": int(p50), "salary_p75": int(p75), "salary_avg": int(avg)},
                    "by_experience": exp_data,
                    "by_position": pos_data
                },
                "data_source": "前程无忧实时数据"
            }
            _stats_cache[key] = (now, result)
            return result

        # 数据库无数据时使用薪酬基准数据
        stats = _salary_fetcher.fetch_statistics(industry, city)
        meta = stats.pop("_meta", {})
        result = {
            "industry": industry,
            "city": city,
            "sample_count": meta.get("sample_count", 1000),
            "statistics": stats,
            "data_source": meta.get("data_source", "行业薪酬基准数据库"),
            "source_description": meta.get("source_description", ""),
        }
        _stats_cache[key] = (now, result)
        return result

    except Exception as e:
        stats = _salary_fetcher.fetch_statistics(industry, city)
        meta = stats.pop("_meta", {})
        result = {
            "industry": industry,
            "city": city,
            "sample_count": meta.get("sample_count", 1000),
            "statistics": stats,
            "data_source": meta.get("data_source", "行业薪酬基准数据库"),
            "source_description": meta.get("source_description", ""),
        }
        _stats_cache[key] = (now, result)
        return result
    finally:
        if conn:
            conn.close()


@app.get("/api/recruitment/list")
def get_recruitment_list(city: str = "北京", limit: int = 50):
    """获取原始招聘数据列表"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT * FROM recruitment_info
            WHERE city = %s
            ORDER BY crawl_time DESC
            LIMIT %s
        """, (city, limit))
        rows = cursor.fetchall()
        return {"count": len(rows), "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            conn.close()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ===== AI 对话与报告端点 =====

@app.post("/api/ai/chat")
async def ai_chat(req: dict):
    """AI 对话：用户用自然语言提问薪资问题"""
    try:
        from ai import ChatEngine, AIConfig
        from ai.schemas import ChatRequest

        config = AIConfig()
        if not config.is_configured:
            return {"reply": "AI 服务未配置，请在 .env 中设置 DEEPSEEK_API_KEY"}

        chat_request = ChatRequest(
            message=req.get("message", ""),
            history=req.get("history", []),
            industry=req.get("industry", "互联网"),
            city=req.get("city", "北京"),
        )
        engine = ChatEngine(config)
        response = await engine.chat(chat_request)
        return {"reply": response.reply}
    except Exception as e:
        return {"reply": f"AI 对话请求失败：{str(e)}"}


@app.post("/api/ai/report")
async def ai_report(req: dict):
    """AI 报告生成：生成行业薪资分析报告"""
    try:
        from ai import ReportGenerator, AIConfig
        from ai.schemas import ReportRequest

        config = AIConfig()
        if not config.is_configured:
            return {"report": "AI 服务未配置，请在 .env 中设置 DEEPSEEK_API_KEY",
                    "industry": req.get("industry", ""), "city": req.get("city", ""),
                    "sample_count": 0}

        report_request = ReportRequest(
            industry=req.get("industry", "互联网"),
            city=req.get("city", "北京"),
        )
        generator = ReportGenerator(config)
        response = await generator.generate(report_request)
        return {
            "report": response.report,
            "industry": response.industry,
            "city": response.city,
            "sample_count": response.sample_count,
        }
    except Exception as e:
        return {"report": f"报告生成失败：{str(e)}",
                "industry": req.get("industry", ""), "city": req.get("city", ""),
                "sample_count": 0}


# ===== Agent 智能体端点 =====

@app.get("/api/agent/tools")
def list_agent_tools():
    """列出 Agent 可用的所有工具"""
    if not orchestrator:
        return {"status": "unconfigured", "tools": []}
    return {
        "status": "ready",
        "count": len(orchestrator.registry),
        "tools": orchestrator.registry.list_tools(),
    }


@app.get("/api/agent/status")
def agent_status():
    """获取 Agent 状态"""
    if not orchestrator:
        return {"status": "unconfigured"}
    return {
        "status": "ready",
        "orchestrator": orchestrator.get_status(),
        "metrics": global_metrics.summary(),
    }


@app.post("/api/agent/chat")
async def agent_chat(req: dict):
    """Agent 对话 — 统一入口（支持简单查询和分析）"""
    if not orchestrator:
        return {"reply": "Agent 服务未配置，请在 .env 中设置 DEEPSEEK_API_KEY",
                "status": "unconfigured"}

    try:
        message = sanitize_input(req.get("message", ""), 4000)

        # 输入检查
        guard = InputGuard.check(message)
        if not guard["passed"]:
            return {"reply": "输入包含不安全内容，请重新描述您的问题。",
                    "status": "rejected"}

        metadata = {
            "industry": req.get("industry", "互联网"),
            "city": req.get("city", "北京"),
        }

        trace = TraceContext()
        trace.start(metadata)

        # 更新工作记忆
        orchestrator.working.update_query(metadata["industry"], metadata["city"])

        result = await orchestrator.process(message)

        # 记录追踪
        report = trace.finish(result.final_answer)
        global_metrics.record(report)

        # 组装步骤详情（供前端展示推理过程）
        steps_detail = []
        for s in result.steps:
            step_info = {"tool": s.tool_name, "args": s.tool_args}
            if s.thought:
                step_info["thought"] = s.thought
            if s.tool_result:
                step_info["result"] = s.tool_result[:300]  # 截断避免过长
            steps_detail.append(step_info)

        return {
            "reply": result.final_answer,
            "status": "success",
            "steps": len(result.steps),
            "steps_detail": steps_detail,
            "trace_id": trace.trace_id,
            "usage": result.token_usage,
        }

    except Exception as e:
        return {"reply": f"处理失败：{str(e)}", "status": "error"}


@app.post("/api/agent/stream")
async def agent_chat_stream(req: dict):
    """Agent 流式对话 — SSE 事件流"""
    if not orchestrator:
        return {"reply": "Agent 服务未配置"}

    message = sanitize_input(req.get("message", ""), 4000)
    metadata = {
        "industry": req.get("industry", "互联网"),
        "city": req.get("city", "北京"),
    }
    orchestrator.working.update_query(metadata["industry"], metadata["city"])

    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        async for event in orchestrator.process_stream(message):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.get("/api/agent/memory")
def agent_memory():
    """查看 Agent 记忆状态（调试用）"""
    if not orchestrator:
        return {"status": "unconfigured"}
    return orchestrator.get_status()


@app.get("/api/agent/metrics")
def agent_metrics():
    """查看 Agent 运行指标"""
    return global_metrics.summary()


@app.get("/api/agent/circuit-breakers")
def circuit_breaker_status():
    """查看熔断器状态"""
    from ai.client import deepseek_cb as ai_cb
    from agent.tools.salary import internal_http_cb as http_cb
    return {
        "circuit_breakers": {
            "deepseek_api": ai_cb.get_status(),
            "mysql": mysql_cb.get_status(),
            "internal_http": http_cb.get_status(),
        }
    }


@app.post("/api/agent/reset")
def agent_reset():
    """重置 Agent 对话状态（含清空会话缓存）"""
    global orchestrator
    if orchestrator:
        orchestrator.short_term.clear()
        orchestrator.working = WorkingMemory()
        orchestrator._query_cache.clear()
    return {"status": "reset"}


# ===== LangGraph Agent 端点 (Phase 5) =============================

_lg_orchestrator = None


def _get_lg_orchestrator():
    """延迟初始化 LangGraph Orchestrator"""
    global _lg_orchestrator
    if _lg_orchestrator is None:
        try:
            from langchain_graph.adapter import LangGraphOrchestrator
            _lg_orchestrator = LangGraphOrchestrator()
        except Exception as e:
            logger = logging.getLogger("app")
            logger.warning(f"LangGraph Orchestrator 初始化失败: {e}")
            return None
    return _lg_orchestrator


@app.post("/api/agent/langgraph/chat")
async def langgraph_chat(req: dict):
    """LangGraph Agent 对话（非流式）"""
    lg = _get_lg_orchestrator()
    if not lg:
        return {"reply": "LangGraph Agent 未就绪", "status": "unconfigured"}

    message = sanitize_input(req.get("message", ""), 4000)
    metadata = {
        "industry": req.get("industry", "互联网"),
        "city": req.get("city", "北京"),
    }
    lg.working.update_query(metadata["industry"], metadata["city"])

    try:
        result = await lg.process(message)
        return {
            "reply": result.final_answer,
            "status": "success",
            "steps": len(result.steps),
            "steps_detail": [
                {"tool": s.tool_name, "args": s.tool_args,
                 "result": s.tool_result[:300] if s.tool_result else ""}
                for s in result.steps
            ],
        }
    except Exception as e:
        return {"reply": f"LangGraph 处理失败: {str(e)}", "status": "error"}


@app.post("/api/agent/langgraph/stream")
async def langgraph_chat_stream(req: dict):
    """LangGraph Agent 流式对话 — SSE 事件流"""
    lg = _get_lg_orchestrator()
    if not lg:
        return {"reply": "LangGraph Agent 未就绪"}

    from sse_starlette.sse import EventSourceResponse

    message = sanitize_input(req.get("message", ""), 4000)
    metadata = {
        "industry": req.get("industry", "互联网"),
        "city": req.get("city", "北京"),
    }
    lg.working.update_query(metadata["industry"], metadata["city"])

    async def event_generator():
        async for event in lg.process_stream(message):
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.get("/api/agent/langgraph/status")
def langgraph_status():
    """LangGraph Agent 状态"""
    lg = _get_lg_orchestrator()
    if not lg:
        return {"status": "unconfigured"}
    return {
        "status": "ready",
        "orchestrator": lg.get_status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
