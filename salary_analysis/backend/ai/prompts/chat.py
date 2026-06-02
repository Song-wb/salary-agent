from .base import SYSTEM_PROMPT

CHAT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
你以对话方式回答用户关于薪资的问题。
回答简洁但数据充分，使用 2-4 个自然段。
如果用户问及具体岗位，优先提供该岗位的薪资数据。
回答结束时可以追问用户是否需要更深入的分析。
"""


def build_chat_prompt(context: str, history: list[dict], user_message: str) -> list[dict]:
    """组装对话消息列表"""
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    if context:
        messages.append({"role": "system", "content": f"当前统计数据：\n{context}"})

    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
