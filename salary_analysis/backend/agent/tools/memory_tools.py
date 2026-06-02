"""记忆工具 — Agent 可调用的记忆检索工具

让 Agent 在 ReAct 循环中自主决定何时检索历史记忆，
实现 Agentic RAG（Agent 自主检索增强生成）。
"""

SEARCH_MEMORY_TOOL = {
    "name": "search_memory",
    "description": "搜索历史分析记忆，找到与当前问题相关的过往分析结论。"
                   "当用户的问题涉及之前讨论过的行业、城市或话题时使用。"
                   "也适用于需要对比不同说法或引用历史数据的场景。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如'北京互联网薪资对比'、'Java开发薪资分析'",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "过滤标签，如 ['互联网', '北京']，缩小搜索范围",
            },
        },
        "required": ["query"],
    },
}


def make_search_handler(memory):
    """创建 search_memory 工具的 handler

    通过闭包捕获 memory 实例（EmbeddingMemory），
    避免在工具函数中引入全局依赖。
    """
    async def search_memory(query: str = "", tags: list[str] | None = None) -> dict:
        """搜索历史记忆，返回相关度最高的结果"""
        results = memory.search(query=query, tags=tags or [], limit=3)
        if not results:
            return {
                "found": False,
                "message": "未找到相关历史记忆",
                "results": [],
            }

        formatted = []
        for m in results:
            formatted.append({
                "content": m.get("content", ""),
                "tags": m.get("tags", []),
                "timestamp": m.get("timestamp", ""),
                "score": m.get("_score", 0),
            })

        return {
            "found": True,
            "count": len(formatted),
            "results": formatted,
        }

    return search_memory
