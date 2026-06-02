"""诊断 LangGraph — 非流式响应完整内容"""
import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 非流式
resp = httpx.post('http://localhost:8000/api/agent/langgraph/chat',
    json={'message': '北京Java开发平均薪资'}, timeout=60)
data = resp.json()
print("=== STATUS ===")
print(json.dumps(data.get('status',''), ensure_ascii=False))
print("\n=== REPLY ===")
print(data.get('reply','')[:1000])
print("\n=== STEPS ===")
for s in data.get('steps_detail', []):
    print(json.dumps(s, ensure_ascii=False)[:300])
