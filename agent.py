import json

from tools import tools, tool_map
from state import AgentState


def call_agent(client, state: AgentState) -> dict:
    """只调用一次模型，返回消息；工具由独立节点执行。"""

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

    # 全程使用 OpenAI 格式的普通字典，直接供 DeepSeek SDK 使用。
    return response.choices[0].message.model_dump(exclude_none=True)


def execute_tools(message: dict) -> dict:
    """只返回本轮增量，不直接修改 State。每个调用都有对应的 tool 消息。"""
    new_messages = []
    history = []
    results = []

    for tool_call in message.get("tool_calls", []):
        function_name = tool_call["function"]["name"]
        arguments = {}
        try:
            # JSON 解析也可能失败，纳入统一错误处理。
            arguments = json.loads(tool_call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            func = tool_map.get(function_name)
            if func is None:
                raise ValueError(f"Unknown tool: {function_name}")
            result = func(**arguments)
            success = True
        except Exception as exc:
            result = f"Tool execution error: {exc}"
            success = False

        print(f"[Tool] {function_name}")
        print(f"[Args] {arguments}")
        print(f"[Result] {result}")

        history.append({
            "tool": function_name,
            "args": arguments if isinstance(arguments, dict) else {},
            "result": str(result),
            "success": success,
        })
        new_messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result),
        })
        results.append({
            "tool": function_name,
            "result": str(result),
            "success": success,
        })

    return {
        "messages": new_messages,
        "tool_history": history,
        "last_action": {"type": "tool", "results": results},
    }
