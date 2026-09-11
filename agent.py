import json

from tools import tools, tool_map
from planner import update_plan_status


MAX_STEPS = 10


def run_agent(client, messages, state):

    for _ in range(MAX_STEPS):

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if not message.tool_calls:

            print("AI：", message.content)

            messages.append(message)

            update_plan_status(client, state,final_answer = message.content)

            print("最终状态：")
            for item in state["plan"]:
                print(item)

            return

        # 必须先记录 assistant 的 tool_calls
        messages.append(message)

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

            # 只记录，不判断 Plan 状态
            state["tool_history"].append({
                "tool": function_name,
                "args": arguments,
                "result": str(result),
                "success": success
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # 一轮工具执行结束后，再统一更新计划
        update_plan_status(client, state)

        print("当前状态：")
        for item in state["plan"]:
            print(item)

    print("AI：达到最大工具调用轮数，任务停止。")