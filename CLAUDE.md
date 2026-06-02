# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A salary analysis platform with AI-powered multi-agent system. Collects real-time recruitment data from 51job, provides salary statistics, AI chat/report generation, and an agent-based analysis engine.

## Quick Start

```bash
# Start all services (MySQL + backend + frontend)
docker-compose up -d

# Collect recruitment data from 51job
python collect.py --city 北京 --pages 3
python collect.py --all --pages 3   # all 10 cities

# Run agent unit tests
python test_agent.py                           # all tests
python test_agent.py --test registry           # single test

# Run backend standalone (without Docker)
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture

### Backend (`salary_analysis/backend/`)

**FastAPI server** on port 8000, three API endpoint groups:
- `/api/salary/*` — statistics and recruitment data queries (MySQL-backed, falls back to benchmark data)
- `/api/ai/*` — direct AI chat and report generation via DeepSeek
- `/api/agent/*` — multi-agent system with tool calling, streaming, and reasoning traces

**AI Layer (`ai/`)**
- `client.py` — AsyncOpenAI wrapper around DeepSeek API (chat + streaming)
- `chat.py` / `report.py` — ChatEngine and ReportGenerator orchestrators
- `config.py` — AIConfig dataclass reading from env (DeepSeek API key, model, temperatures)
- `context.py` — SalaryContext that fetches live stats from the FastAPI endpoints and assembles LLM-friendly text context
- `schemas.py` — Pydantic request/response models
- `prompts/` — System prompt builders for chat and report modes

**Agent System (`agent/`)**
- **ReActAgent** (`core.py`) — the fundamental reasoning loop: LLM thinks, decides tool call, executes, observes, loops
- **Orchestrator** (`orchestrator.py`) — classifies requests (simple/analysis/report/chart/complex), dispatches to specialist agents, manages memory lifecycle
- **ToolRegistry** (`registry.py`) — register, list, and execute tools; provides OpenAI-compatible function calling format
- **Specialists** (`specialists/`) — DataAnalystAgent, ReportWriterAgent, ChartCreatorAgent — each wraps ReActAgent with a role-specific system prompt
- **Tools** (`tools/`) — salary.py (statistics/recruitment queries), analysis.py (city/industry comparison), mcp_tools.py (Word document generation)
- **Memory** (`memory/`) — ShortTermMemory (sliding window + auto-summary), WorkingMemory (current query/task state), VectorMemory (file-based JSON persistence with tag/keyword search)
- **Guardrails** (`guardrails.py`) — InputGuard for prompt injection detection, OutputGuard for answer quality checks
- **Observability** (`observability.py`) — TraceContext per-request tracing, AgentMetrics global counters, structured JSON logging

**Data Layer**
- MySQL with `recruitment_info` (raw job postings) and `salary_statistics` tables
- `fetchers/salary_benchmark.py` — static benchmark data for 50+ positions across 8 industries and 10 cities, with city/industry adjustment coefficients
- Data flow: 51job crawl -> MySQL -> FastAPI endpoints -> direct API responses and agent tool results

**Data Collection (`collect.py`)**
- Scrapes 51job for each city, parses `window.__SEARCH_RESULT__` JSON from HTML
- Writes directly to the Docker MySQL instance
- Supports --cron for scheduled collection

### Frontend (`salary_analysis/frontend/`)

Streamlit app on port 8501, three tabs:
- **Data Dashboard** — stats cards, Plotly charts for experience-salary and position comparison, raw data table
- **AI Chat** — toggle between Agent mode (streaming ReAct with reasoning trace display) and normal chat mode
- **AI Report** — generates structured salary analysis report with markdown download

### Root Files

- `final_verify.py` — utility to read docx evaluation reports
- `Practice/` — SCUT course materials (separate from salary analysis)

## Key Configuration

Environment variables (`.env`): `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, MySQL credentials. Backend defaults to `deepseek-chat` model. Frontend connects to backend at `http://backend:8000` (Docker internal network).
