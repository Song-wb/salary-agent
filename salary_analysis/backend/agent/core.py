"""ReAct Agent Core — 思考→行动→观察 推理循环"""

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator
from openai import AsyncOpenAI

from .registry import ToolRegistry


@dataclass
class AgentStep:
    """Agent 单个推理步骤"""
    thought: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    tool_call_id: str = ""
    timestamp: float = 0.0


@dataclass
class AgentResult:
    """Agent 执行结果"""
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    total_time: float = 0.0
    tool_call_count: int = 0


DEFAULT_SYSTEM_PROMPT = """你是一个专业的薪酬分析智能助手，精通中国各行业薪酬体系。
你需要通过调用工具来获取数据，然后基于数据进行分析和回答。

你可以使用的工具包括：
- query_salary_statistics / query_recruitment_list: 查询实时薪资数据
- compare_cities / compare_industries: 多维度对比分析
- search_memory: 搜索历史分析记录，当用户问题涉及之前讨论过的内容时使用
- create_word_document: 生成 Word 分析报告

核心原则：
1. 始终基于数据说话，不得编造数据
2. 分析要有深度，提供洞察而不仅仅是罗列数字
3. 使用中文回答，薪资单位使用"元/月"
4. 回答结构清晰，包含具体数字和对比分析
5. 当用户问题与历史对话相关时，先调用 search_memory 查找背景信息
6. 获取数据后再进行分析和总结"""


class ReActAgent:
    """ReAct 推理循环 Agent

    核心流程：
    1. 接收用户消息
    2. LLM 思考 → 决定行动（工具调用）或直接回答
    3. 如果调用工具 → 执行工具 → 观察结果 → 回到步骤 2
    4. 如果直接回答 → 结束
    """

    def __init__(self, config, registry: ToolRegistry, reflector=None):
        self.config = config
        self.registry = registry
        if reflector is not None:
            self.reflector = reflector
        else:
            from .reflection.reflector import Reflector
            self.reflector = Reflector()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    async def _execute_with_cache(self, tool_name: str, args: dict,
                                   session_cache: dict | None = None) -> Any:
        """执行工具，带会话级缓存（L3）

        在 registry.execute 之前检查 session_cache，
        仅对 query_salary_statistics 生效（确定性查询）。
        """
        if session_cache is not None and tool_name == "query_salary_statistics":
            key = (args.get("industry", "互联网"), args.get("city", "北京"))
            if key in session_cache:
                return session_cache[key]

        raw = await self.registry.execute(tool_name, args)

        if session_cache is not None and tool_name == "query_salary_statistics":
            if isinstance(raw, dict) and "industry" in raw and "city" in raw:
                key = (raw["industry"], raw["city"])
                session_cache[key] = raw

        return raw

    async def run(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context_text: str = "",
        max_steps: int = 10,
        session_cache: dict | None = None,
    ) -> AgentResult:
        """执行完整 ReAct 循环"""
        start_time = time.time()
        result = AgentResult()
        messages = self._build_messages(user_message, system_prompt, context_text)
        tools = self.registry.get_openai_tools()
        continue_loop = True
        reflection_depth = 0

        while continue_loop and len(result.steps) < max_steps:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.chat_temperature,
                max_tokens=self.config.max_tokens,
                tools=tools if tools else None,
                stream=False,
            )

            choice = response.choices[0]
            result.token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # 无工具调用 → 最终回答
            if not choice.finish_reason == "tool_calls":
                result.final_answer = choice.message.content or ""

                # [Point B] 最终回答质量检查
                if reflection_depth < self.reflector.config.max_reflection_depth:
                    ref_result = self.reflector.check_final_answer(result.final_answer)
                    if not ref_result.passed:
                        messages.extend(ref_result.corrective_messages)
                        reflection_depth += 1
                        continue  # 不增加 step count，回去 LLM 修正

                break

            # 处理工具调用
            if choice.message.tool_calls:
                # 先将 assistant 的消息（含所有 tool_calls）追加一次
                messages.append(choice.message)

                # 判断是否可并行执行（所有工具都是只读查询）
                if self._is_read_only(choice.message.tool_calls):
                    steps = await self._execute_tools_parallel(
                        choice.message.tool_calls, choice.message.content or "",
                        session_cache=session_cache,
                    )
                    for step in steps:
                        result.steps.append(step)
                        result.tool_call_count += 1
                        messages.append({
                            "role": "tool",
                            "tool_call_id": step.tool_call_id,
                            "content": step.tool_result,
                        })

                    # [Point C] 反射检查：工具错误、空结果、数据矛盾
                    if reflection_depth < self.reflector.config.max_reflection_depth:
                        ref_result = self.reflector.check_tool_results(result.steps)
                        if not ref_result.passed:
                            messages.extend(ref_result.corrective_messages)
                            reflection_depth += 1
                            continue  # 不增加 step count，回去 LLM 修正
                else:
                    for tc in choice.message.tool_calls:
                        step = AgentStep(timestamp=time.time())
                        step.thought = choice.message.content or ""

                        fn = tc.function
                        step.tool_name = fn.name
                        try:
                            step.tool_args = json.loads(fn.arguments)
                        except json.JSONDecodeError:
                            step.tool_args = {"raw": fn.arguments}

                        # 执行工具（带 L3 会话缓存）
                        raw_result = await self._execute_with_cache(
                            fn.name, step.tool_args, session_cache
                        )
                        step.tool_result = self._format_result(raw_result)

                        result.steps.append(step)
                        result.tool_call_count += 1

                        # 每个 tool call 对应一个 tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": step.tool_result,
                        })

                    # [Point D] 反射检查：工具错误、空结果、数据矛盾
                    if reflection_depth < self.reflector.config.max_reflection_depth:
                        ref_result = self.reflector.check_tool_results(result.steps)
                        if not ref_result.passed:
                            messages.extend(ref_result.corrective_messages)
                            reflection_depth += 1
                            continue  # 不增加 step count，回去 LLM 修正

                # 检查是否需要继续
                if result.tool_call_count >= max_steps:
                    continue_loop = False

        result.total_time = time.time() - start_time
        if not result.final_answer and result.steps:
            # 达到最大步数但仍无最终回答，让 LLM 总结
            messages.append({
                "role": "user",
                "content": "请基于以上数据，给出最终的分析总结。"
            })
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.chat_temperature,
                max_tokens=self.config.max_tokens,
            )
            result.final_answer = response.choices[0].message.content or ""

            # [Point E] 反射检查：总结生成的最终回答（已在循环外，直接修正）
            if reflection_depth < self.reflector.config.max_reflection_depth:
                ref_result = self.reflector.check_final_answer(result.final_answer)
                if not ref_result.passed:
                    messages.extend(ref_result.corrective_messages)
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.chat_temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    result.final_answer = response.choices[0].message.content or ""

        return result

    async def run_stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context_text: str = "",
        max_steps: int = 10,
        session_cache: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式 ReAct 循环，逐事件产出"""
        start_time = time.time()
        messages = self._build_messages(user_message, system_prompt, context_text)
        tools = self.registry.get_openai_tools()

        yield {"type": "start", "message": user_message}

        step_count = 0
        reflection_depth = 0
        steps_history: list = []  # 用于反射检查的步骤历史
        final_content = ""

        while step_count < max_steps:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.chat_temperature,
                max_tokens=self.config.max_tokens,
                tools=tools if tools else None,
                stream=False,
            )

            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if response.usage else {}

            # 无工具调用 → 最终回答
            if choice.finish_reason != "tool_calls":
                content = choice.message.content or ""
                final_content = content
                yield {"type": "text", "content": content}

                # [Point B 流式] 最终回答质量检查
                if reflection_depth < self.reflector.config.max_reflection_depth:
                    ref_result = self.reflector.check_final_answer(content)
                    if not ref_result.passed:
                        yield {
                            "type": "reflection",
                            "issues": [
                                {"type": i.check_type, "severity": i.severity,
                                 "message": i.message, "details": i.details}
                                for i in ref_result.issues
                            ],
                            "reflection_depth": reflection_depth + 1,
                            "max_depth": self.reflector.config.max_reflection_depth,
                        }
                        messages.extend(ref_result.corrective_messages)
                        reflection_depth += 1
                        continue  # 不增加 step count，回去 LLM 修正

                yield {"type": "done", "content": content, "steps": step_count,
                       "token_usage": usage, "total_time": time.time() - start_time}
                return

            # 处理工具调用
            thought = choice.message.content or ""
            if thought:
                yield {"type": "thought", "content": thought}

            # 先将 assistant 消息（含所有 tool_calls）追加一次
            messages.append(choice.message)

            if self._is_read_only(choice.message.tool_calls):
                # 并行执行
                yield {"type": "tool_parallel_start",
                       "count": len(choice.message.tool_calls),
                       "names": [tc.function.name for tc in choice.message.tool_calls]}

                steps = await self._execute_tools_parallel(
                    choice.message.tool_calls, thought,
                    session_cache=session_cache,
                )
                for step in steps:
                    step_count += 1
                    yield {"type": "tool_call", "name": step.tool_name,
                           "args": step.tool_args}
                    yield {"type": "tool_result", "name": step.tool_name,
                           "result": step.tool_result}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": step.tool_call_id,
                        "content": step.tool_result,
                    })
                    steps_history.append(step)

                # [Point C 流式] 反射检查
                if reflection_depth < self.reflector.config.max_reflection_depth:
                    ref_result = self.reflector.check_tool_results(steps_history)
                    if not ref_result.passed:
                        yield {
                            "type": "reflection",
                            "issues": [
                                {"type": i.check_type, "severity": i.severity,
                                 "message": i.message, "details": i.details}
                                for i in ref_result.issues
                            ],
                            "reflection_depth": reflection_depth + 1,
                            "max_depth": self.reflector.config.max_reflection_depth,
                        }
                        messages.extend(ref_result.corrective_messages)
                        reflection_depth += 1
                        continue
            else:
                for tc in choice.message.tool_calls:
                    fn = tc.function
                    step_count += 1
                    try:
                        args = json.loads(fn.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": fn.arguments}

                    yield {"type": "tool_call", "name": fn.name, "args": args}

                    raw_result = await self._execute_with_cache(fn.name, args, session_cache)
                    formatted = self._format_result(raw_result)

                    yield {"type": "tool_result", "name": fn.name, "result": formatted}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": formatted,
                    })

                    # 记录到 steps_history
                    tmp_step = AgentStep(
                        tool_name=fn.name,
                        tool_args=args,
                        tool_result=formatted,
                    )
                    steps_history.append(tmp_step)

                # [Point D 流式] 反射检查
                if reflection_depth < self.reflector.config.max_reflection_depth:
                    ref_result = self.reflector.check_tool_results(steps_history)
                    if not ref_result.passed:
                        yield {
                            "type": "reflection",
                            "issues": [
                                {"type": i.check_type, "severity": i.severity,
                                 "message": i.message, "details": i.details}
                                for i in ref_result.issues
                            ],
                            "reflection_depth": reflection_depth + 1,
                            "max_depth": self.reflector.config.max_reflection_depth,
                        }
                        messages.extend(ref_result.corrective_messages)
                        reflection_depth += 1
                        continue

        # 达到最大步数，总结
        messages.append({
            "role": "user",
            "content": "请基于以上数据，给出最终的分析总结。"
        })
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.chat_temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        yield {"type": "text", "content": content}

        # [Point E 流式] 反射检查：总结生成的最终回答
        if reflection_depth < self.reflector.config.max_reflection_depth:
            ref_result = self.reflector.check_final_answer(content)
            if not ref_result.passed:
                yield {
                    "type": "reflection",
                    "issues": [
                        {"type": i.check_type, "severity": i.severity,
                         "message": i.message, "details": i.details}
                        for i in ref_result.issues
                    ],
                    "reflection_depth": reflection_depth + 1,
                    "max_depth": self.reflector.config.max_reflection_depth,
                }
                messages.extend(ref_result.corrective_messages)
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.chat_temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.choices[0].message.content or ""
                yield {"type": "text", "content": content}

        yield {"type": "done", "content": content, "steps": step_count,
               "token_usage": {}, "total_time": time.time() - start_time}

    @staticmethod
    def _is_read_only(tool_calls: list) -> bool:
        """判断一组工具调用是否全是只读查询（可并行）"""
        READ_ONLY_TOOLS = {
            "query_salary_statistics", "query_recruitment_list",
            "compare_cities", "compare_industries",
        }
        return all(tc.function.name in READ_ONLY_TOOLS for tc in tool_calls)

    async def _execute_tools_parallel(self, tool_calls: list, thought: str,
                                       session_cache: dict | None = None) -> list:
        """并行执行多个工具调用，返回 AgentStep 列表（或兼容的 dict 列表）"""

        async def execute_one(tc):
            fn = tc.function
            try:
                args = json.loads(fn.arguments)
            except json.JSONDecodeError:
                args = {"raw": fn.arguments}

            raw_result = await self._execute_with_cache(fn.name, args, session_cache)
            step = AgentStep(timestamp=time.time())
            step.thought = thought
            step.tool_name = fn.name
            step.tool_args = args
            step.tool_result = self._format_result(raw_result)
            # 将 tool_call_id 挂载到 step 上供后续追加 messages
            step.tool_call_id = tc.id
            return step

        results = await asyncio.gather(
            *[execute_one(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        steps = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                tc = tool_calls[i]
                fn = tc.function
                step = AgentStep(timestamp=time.time())
                step.thought = thought
                step.tool_name = fn.name
                step.tool_args = {}
                step.tool_result = json.dumps({"error": f"工具 [{fn.name}] 并行执行失败: {str(res)}"})
                step.tool_call_id = tc.id
                steps.append(step)
            else:
                steps.append(res)
        return steps

    def _build_messages(self, user_message: str, system_prompt: str | None,
                        context_text: str) -> list[dict]:
        """组装消息列表"""
        sp = system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": sp}]
        if context_text:
            messages.append({"role": "system", "content": f"当前上下文数据：\n{context_text}"})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _format_result(raw) -> str:
        """格式化工具执行结果为字符串"""
        if isinstance(raw, str):
            return raw[:3000]  # 截断过长结果
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False, default=str)[:3000]
        return str(raw)[:3000]
