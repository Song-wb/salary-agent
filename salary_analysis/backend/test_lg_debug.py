"""诊断 LangGraph Agent 流式响应 — 打印所有事件"""
import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with httpx.Client() as client:
    with client.stream('POST', 'http://localhost:8000/api/agent/langgraph/stream',
            json={'message': '北京Java开发平均薪资'}, timeout=60) as resp:
        for line in resp.iter_lines():
            if line and line.startswith('data: '):
                event = json.loads(line[6:])
                t = event.get('type', '')
                print(f'[{t}]', json.dumps(event, ensure_ascii=False, default=str)[:500])
