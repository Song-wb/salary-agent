import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import json

# 设置页面配置
st.set_page_config(
    page_title="行业薪酬洞察AI智能体",
    page_icon="💼",
    layout="wide"
)

API_BASE = "http://backend:8000"

# 获取数据（定义在前面）
@st.cache_data
def fetch_data(industry, city):
    try:
        response = requests.get(
            f"{API_BASE}/api/salary/statistics",
            params={"industry": industry, "city": city}
        )
        return response.json()
    except Exception as e:
        return {
            "industry": industry, "city": city, "sample_count": 1256,
            "data_source": "模拟数据",
            "statistics": {
                "overall": {"salary_p25": 15000, "salary_p50": 22000, "salary_p75": 32000, "salary_avg": 24500},
                "by_experience": [
                    {"experience": "0-1年", "avg_salary": 12000, "count": 156},
                    {"experience": "1-3年", "avg_salary": 18000, "count": 342},
                    {"experience": "3-5年", "avg_salary": 25000, "count": 428},
                    {"experience": "5年+", "avg_salary": 32000, "count": 330}
                ],
                "by_position": [
                    {"position": "Java开发", "avg_salary": 26000, "count": 234},
                    {"position": "产品经理", "avg_salary": 24000, "count": 189},
                    {"position": "前端开发", "avg_salary": 23000, "count": 215},
                    {"position": "数据分析师", "avg_salary": 21000, "count": 168},
                    {"position": "运维工程师", "avg_salary": 20000, "count": 145}
                ]
            }
        }

def call_agent(message, industry, city):
    """调用 Agent 对话接口"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/agent/chat",
            json={"message": message, "industry": industry, "city": city},
            timeout=120
        )
        return resp.json()
    except Exception as e:
        return {"reply": f"请求失败：{str(e)}", "status": "error"}

def call_agent_tools():
    """获取 Agent 工具列表"""
    try:
        resp = requests.get(f"{API_BASE}/api/agent/tools", timeout=5)
        return resp.json()
    except:
        return {"tools": []}

def call_agent_metrics():
    """获取 Agent 指标"""
    try:
        resp = requests.get(f"{API_BASE}/api/agent/metrics", timeout=5)
        return resp.json()
    except:
        return {}

# 标题
st.title("💼 行业薪酬洞察AI智能体")

# 筛选条件（放在侧边栏，节省主区域空间）
with st.sidebar:
    st.header("⚙️ 筛选条件")
    industry = st.selectbox("选择行业", ["互联网", "金融", "制造业", "医疗健康", "教育", "房地产", "消费品", "物流"])
    city = st.selectbox("选择城市", ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "南京", "西安", "重庆"])

    st.divider()
    st.subheader("🤖 Agent 状态")
    tools = call_agent_tools()
    if tools.get("status") == "ready":
        st.success(f"Agent 就绪，{tools.get('count', 0)} 个工具可用")
        with st.expander("查看可用工具"):
            for t in tools.get("tools", []):
                st.caption(f"**{t['name']}**: {t['description'][:80]}...")
    else:
        st.warning("Agent 未配置（需设置 API Key）")

    metrics = call_agent_metrics()
    if metrics:
        st.caption(f"会话数: {metrics.get('total_sessions', 0)}")
        st.caption(f"Token: {metrics.get('total_tokens', 0)}")
        st.caption(f"工具调用: {metrics.get('total_tool_calls', 0)}")

data = fetch_data(industry, city)

# 主区域使用 Tab 切换
tab1, tab2, tab3 = st.tabs(["📊 数据看板", "💬 AI 对话", "🤖 AI 报告"])

# ===== Tab1: 数据看板 =====
with tab1:
    st.header("📊 薪资概览")
    stats = data["statistics"]["overall"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("25分位薪资", f"{stats['salary_p25']/1000:.0f}K")
    col2.metric("中位数薪资", f"{stats['salary_p50']/1000:.0f}K")
    col3.metric("75分位薪资", f"{stats['salary_p75']/1000:.0f}K")
    col4.metric("平均薪资", f"{stats['salary_avg']/1000:.0f}K")

    st.header("📈 经验-薪资曲线")
    if data["statistics"]["by_experience"]:
        exp_df = pd.DataFrame(data["statistics"]["by_experience"])
        fig1 = px.bar(exp_df, x="experience", y="avg_salary",
                      title="各经验段平均薪资",
                      labels={"avg_salary": "平均薪资(元)", "experience": "经验要求"})
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("暂无经验薪资数据")

    st.header("🏢 岗位薪资对比")
    if data["statistics"]["by_position"]:
        pos_df = pd.DataFrame(data["statistics"]["by_position"])
        fig2 = px.bar(pos_df, x="position", y="avg_salary",
                      title="各岗位平均薪资",
                      labels={"avg_salary": "平均薪资(元)", "position": "岗位名称"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无岗位薪资数据")

    st.header("📋 原始招聘数据")
    if st.checkbox("显示原始数据列表", value=False):
        try:
            response = requests.get(
                f"{API_BASE}/api/recruitment/list",
                params={"city": city, "limit": 50}
            )
            result = response.json()
            if "data" in result and result["data"]:
                df = pd.DataFrame(result["data"])
                df["salary_range"] = df.apply(lambda x: f"{x['salary_min']/1000:.0f}K-{x['salary_max']/1000:.0f}K", axis=1)
                st.dataframe(df[["position_name", "company_name", "salary_range", "experience", "education", "publish_time"]])
            else:
                st.info("暂无原始数据，请先抓取数据")
        except Exception as e:
            st.info("无法获取原始数据（后端可能未启动）")

    st.header("📝 数据说明")
    st.write(f"**行业**：{data['industry']} 行业，{data['city']} 地区")
    st.write(f"**样本数量**：{data['sample_count']} 条招聘信息")
    ds = data.get("data_source", "未知")
    if "基准" in ds or "数据库" in ds:
        st.success(f"✅ **数据源**：{ds}")
    elif "模拟" in ds or "降级" in ds:
        st.warning(f"⚠️ **数据源**：{ds}")
    else:
        st.info(f"📊 **数据源**：{ds}")
    if "source_description" in data and data["source_description"]:
        st.caption(data["source_description"])
    st.write("**统计周期**：近一个月")

# ===== Tab2: AI 对话 =====
with tab2:
    st.header("💬 AI 薪资顾问")

    # Agent 模式切换
    col_mode, col_status = st.columns([1, 3])
    with col_mode:
        agent_mode = st.toggle("🧠 Agent 模式（推理+工具调用）", value=True,
                               help="开启后 AI 将使用 ReAct 推理循环，自主调用工具获取数据并分析")
    with col_status:
        if agent_mode:
            st.info("Agent 模式：AI 会自主思考、调用工具获取数据、多步分析后回答")
        else:
            st.caption("普通模式：AI 基于已有上下文直接回答")

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "steps_detail" in msg and msg["steps_detail"]:
                with st.expander(f"查看推理过程 ({len(msg['steps_detail'])} 步)", expanded=False):
                    for i, step in enumerate(msg["steps_detail"], 1):
                        st.markdown(f"**步骤 {i}**")
                        if step.get("thought"):
                            st.caption(f"思考: {step['thought'][:200]}")
                        st.caption(f"工具: {step.get('tool', '?')}")
                        st.code(str(step.get("args", {})), language="json")
                        if step.get("result"):
                            st.caption(f"结果: {step['result'][:200]}")

# ===== 全局输入框（放在 tabs 外部） =====
if prompt := st.chat_input("💬 提问：例如「北京Java开发薪资水平如何？」"):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 处理中
        with st.chat_message("assistant"):
            if agent_mode:
                # Agent 模式 — 流式返回
                text_placeholder = st.empty()
                tool_container = st.container()
                status_indicator = st.status("🤖 Agent 正在思考...", expanded=True)

                full_text = ""
                steps_detail = []
                steps_count = 0
                usage = {}
                tool_idx = 0
                parallel_tool_count = 0
                completed_tool_count = 0

                try:
                    with requests.post(
                        f"{API_BASE}/api/agent/stream",
                        json={"message": prompt, "industry": industry, "city": city},
                        stream=True,
                        timeout=120
                    ) as resp:
                        for raw_line in resp.iter_lines(decode_unicode=True):
                            if not raw_line or not raw_line.startswith("data: "):
                                continue
                            event = json.loads(raw_line[6:])
                            et = event.get("type")

                            if et == "task_type":
                                status_indicator.write(f"📋 任务类型: {event['task_type']}")

                            elif et == "thought":
                                status_indicator.write(f"🤔 {event.get('content', '')}")

                            elif et == "tool_parallel_start":
                                parallel_tool_count = event.get("count", 0)
                                status_indicator.write(
                                    f"⚡ 并行执行 {parallel_tool_count} 个工具..."
                                )

                            elif et == "tool_call":
                                tool_idx += 1
                                steps_detail.append({
                                    "tool": event["name"],
                                    "args": event.get("args", {}),
                                    "thought": event.get("thought", ""),
                                })
                                if parallel_tool_count > 0:
                                    status_indicator.write(
                                        f"🔧 工具 {tool_idx}/{parallel_tool_count}: `{event['name']}()`"
                                    )
                                else:
                                    status_indicator.write(f"🔧 步骤 {tool_idx}: `{event['name']}()`")
                                if event.get("args"):
                                    with tool_container:
                                        st.code(json.dumps(event["args"], ensure_ascii=False), language="json")

                            elif et == "tool_start":
                                tool_idx += 1
                                steps_detail.append({
                                    "tool": event["tool"],
                                    "args": event.get("args", {}),
                                    "thought": event.get("thought", ""),
                                })
                                status_indicator.write(
                                    f"🔧 启动: `{event['tool']}()`"
                                )
                                if event.get("args"):
                                    with tool_container:
                                        st.code(json.dumps(event["args"], ensure_ascii=False), language="json")

                            elif et == "tool_end":
                                completed_tool_count += 1
                                tool_name = event.get("tool", "工具")
                                duration = event.get("duration_s", 0)
                                status_text = f"✅ {tool_name} 完成 ({duration:.1f}s)"
                                if completed_tool_count > 0 and parallel_tool_count > 0:
                                    status_text += f" [{completed_tool_count}/{parallel_tool_count}]"
                                status_indicator.write(status_text)
                                if steps_detail:
                                    steps_detail[-1]["result"] = str(event.get("result_summary", ""))

                            elif et == "tool_error":
                                status_indicator.write(
                                    f"❌ {event.get('tool', '工具')}: {event.get('error', '')}"
                                )

                            elif et == "tool_result":
                                status_indicator.write(f"✅ {event.get('name', '工具')} 返回结果")
                                if steps_detail:
                                    steps_detail[-1]["result"] = str(event.get("result", ""))[:300]

                            elif et == "text":
                                full_text += event.get("content", "")
                                text_placeholder.markdown(full_text + "▌")

                            elif et == "done":
                                full_text = event.get("content", full_text)
                                text_placeholder.markdown(full_text)
                                steps_count = event.get("steps", len(steps_detail))
                                usage = event.get("usage", {})
                                status_indicator.update(
                                    label=f"✅ Agent 完成 ({steps_count} 步)",
                                    state="complete", expanded=False
                                )

                except Exception as e:
                    full_text = f"请求失败：{str(e)}"
                    text_placeholder.markdown(full_text)
                    status_indicator.update(label="❌ 处理失败", state="error")

                # 显示推理步骤详情
                if steps_detail:
                    with st.expander(f"查看推理过程 ({len(steps_detail)} 步)", expanded=False):
                        for i, step in enumerate(steps_detail, 1):
                            st.markdown(f"**步骤 {i}:** `{step.get('tool', '?')}`")
                            if step.get("thought"):
                                st.caption(f"思考: {step['thought'][:300]}")
                            st.code(str(step.get("args", {})), language="json")
                            if step.get("result"):
                                st.caption(f"结果: {step['result'][:300]}")

                # Agent 元信息
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"推理步数: {steps_count}")
                with cols[1]:
                    st.caption(f"Token: {usage.get('total_tokens', 'N/A')}")
                with cols[2]:
                    if steps_detail:
                        st.caption(f"工具调用: {steps_count}")

                # 保存到 session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "steps": steps_count,
                    "steps_detail": steps_detail,
                })

            else:
                # 普通模式（使用原有 AI 对话接口）
                with st.spinner("思考中..."):
                    try:
                        history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ]
                        resp = requests.post(
                            f"{API_BASE}/api/ai/chat",
                            json={"message": prompt, "history": history,
                                  "industry": industry, "city": city},
                            timeout=60
                        )
                        reply = resp.json().get("reply", "抱歉，暂时无法回答")
                    except Exception as e:
                        reply = f"请求失败：{str(e)}"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

# ===== Tab3: AI 报告 =====
with tab3:
    st.header("🤖 AI 分析报告")
    st.caption(f"基于 {industry} 行业 {city} 地区的统计数据，由 DeepSeek 自动生成分析报告")

    if st.button("📄 生成报告", use_container_width=True):
        with st.spinner("AI 正在生成报告（约 10-30 秒）..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/ai/report",
                    json={"industry": industry, "city": city},
                    timeout=60
                )
                report_data = resp.json()
                if "report" in report_data:
                    st.markdown(report_data["report"])
                    st.download_button(
                        "⬇️ 下载报告 (.md)",
                        report_data["report"],
                        file_name=f"{industry}_{city}_薪资分析报告.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("报告生成失败")
            except Exception as e:
                st.error(f"报告生成失败：{str(e)}")
