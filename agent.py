import json

from tools import tools, tool_map


def run_agent_step(client, state):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=state["messages"],
        tools=tools
    )

    message = response.choices[0].message


    # 没有工具调用，说明 LLM 已经给出最终回答
    if not message.tool_calls:

        print("AI：", message.content)

        state["messages"].append(message)

        return {
            "type": "final",
            "content": message.content
        }


    # 必须先记录 assistant 的 tool_calls
    state["messages"].append(message)

    results = []


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