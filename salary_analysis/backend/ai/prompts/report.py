from .base import SYSTEM_PROMPT

REPORT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
你是资深薪酬分析顾问，生成结构化的行业薪资分析报告。
报告必须包含以下 Markdown 章节：

## 1. 市场概况
行业背景、样本量、数据来源、整体市场判断

## 2. 薪资分布分析
25分位/中位数/75分位/平均薪资，薪资区间分布解读

## 3. 岗位薪资对比
Top 岗位排序，薪酬差距分析，热门岗位供需情况

## 4. 经验-薪资关系
各经验段薪资数据，成长曲线点评，薪资增长率计算

## 5. 趋势与建议
市场判断，求职建议，薪酬策略建议

每个章节必须有具体数据支撑，避免空泛描述。不要编造不在上下文中的数据。
报告格式为 Markdown，适合直接展示。
"""


def build_report_prompt(context: str, industry: str, city: str) -> list[dict]:
    """组装报告生成消息列表"""
    messages = [{"role": "system", "content": REPORT_SYSTEM_PROMPT}]

    if context:
        messages.append({"role": "system", "content": f"统计数据：\n{context}"})

    messages.append({
        "role": "user",
        "content": f"请为{industry}行业{city}地区生成一份薪资分析报告"
    })
    return messages
