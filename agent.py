import json

from tools import tools, tool_map
from state import AgentState, LastAction, ToolStepResult


def run_agent_step(client, state: AgentState) -> LastAction:

    # 每一轮都临时注入最新计划，
    # 不把旧计划不断保存进 state["messages"]
    agent_messages = state["messages"].copy()

    plan_text = json.dumps(
        state["plan"],
        ensure_ascii=False,
        indent=2
    )

    agent_messages.append({
        "role": "system",
        "content": f"""
当前最新任务计划：

{plan_text}

请根据最新计划继续执行任务。

注意：

1. 不要重复执行 status 为 done 的步骤。
2. 优先完成 status 为 pending 的步骤。
3. failed 表示该步骤已经确认失败，不要无意义重复。
4. 如果需要工具，调用合适的工具。
5. 如果任务已经全部完成，直接给出最终回答。
"""
    })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=agent_messages,
        tools=tools
    )

    message = response.choices[0].message


    # 没有工具调用，说明 LLM 已经给出最终回答
    if not message.tool_calls:

        content = message.content or ""

        print("AI：", content)

        state["messages"].append(message)

        return {
            "type": "final",
            "content": content
        }


    # 必须先记录 assistant 的 tool_calls
    state["messages"].append(message)

    results: list[ToolStepResult] = []


    # 一次 Agent Step 可能包含多个 Tool Call
    for tool_call in message.tool_calls:

        function_name = tool_call.function.name

        arguments = json.loads(
            tool_call.function.arguments
        )

        print(f"[Tool] {function_name}")
        print(f"[Args] {arguments}")

        func = tool_map.get(function_name)


        if func is None:

            result = (
                f"Error: tool "
                f"'{function_name}' does not exist."
            )

            success = False

        else:

            try:
                result = func(**arguments)
                success = True

            except Exception as e:
                result = f"Tool execution error: {e}"
                success = False


        print(f"[Result] {result}")


        # 给 Python Workflow 保存结构化工具历史
        state["tool_history"].append({
            "tool": function_name,
            "args": arguments,
            "result": str(result),
            "success": success
        })


        # 给 LLM 保存工具结果
        state["messages"].append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })


        # 记录这一轮 Action 的结果
        results.append({
            "tool": function_name,
            "result": str(result),
            "success": success
        })


    return {
        "type": "tool",
        "results": results
    }
