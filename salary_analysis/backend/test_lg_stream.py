"""测试 LangGraph Agent 流式响应"""
import httpx, json, sys

# Fix Windows GBK encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore

with httpx.Client() as client:
    with client.stream('POST', 'http://localhost:8000/api/agent/langgraph/stream',
            json={'message': '北京Java开发平均薪资'}, timeout=60) as resp:
        for line in resp.iter_lines():
            if line and line.startswith('data: '):
                event = json.loads(line[6:])
                t = event.get('type', '')
                if t == 'tool_result':
                    print(f'[tool_result] {event.get("name","")}: {event.get("result","")[:200]}')
                elif t == 'tool_call':
                    args = json.dumps(event.get("args",{}), ensure_ascii=False)
                    print(f'[tool_call] {event.get("name","")}({args})')
                elif t == 'text':
                    print(f'[text] {event.get("content","")[:100]}')
                elif t == 'tool_error':
                    print(f'[tool_error] {event.get("error","")}')
                elif t == 'tool_parallel_start':
                    print(f'[parallel_start] {event.get("count")} tools: {event.get("names")}')
                elif t == 'tool_end':
                    print(f'[tool_end] {event.get("tool")} ({event.get("duration_s")}s)')
                elif t == 'thought':
                    print(f'[thought] {event.get("content","")[:100]}')
                elif t == 'task_type':
                    print(f'[task_type] {event.get("task_type")}')
                elif t == 'reflection':
                    print(f'[reflection] depth={event.get("reflection_depth")}/{event.get("max_depth")}')
                elif t == 'done':
                    print(f'[done] steps={event.get("steps",0)}, time={event.get("total_time",0):.1f}s')
                    # print final content
                    content = event.get("content", "")
                    if content:
                        print(f"\n=== Final Answer ===\n{content[:500]}")
                    break
