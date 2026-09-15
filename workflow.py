from planner import make_plan, update_plan_status, replan
from agent import run_agent_step


MAX_STEPS = 10
MAX_RETRIES = 2

def retry_node(state):

    state["retry_count"] += 1

    state["messages"].append({
        "role": "user",
        "content": f"""
上一轮工具执行失败。

这是第 {state["retry_count"]} 次重试。

请根据工具返回的错误信息，
重新尝试完成当前任务。
如果原参数有问题，可以调整参数。
"""
    })

    return state

def planner_node(client, state):

    plan = make_plan(
        client,
        state["user_input"]
    )

    state["plan"] = plan

    return state


def action_node(client, state):

    result = run_agent_step(
        client,
        state
    )

    state["last_action"] = result

    return state

def replan_node(client, state):

    new_plan = replan(
        client,
        state
    )

    state["plan"] = new_plan
    state["retry_count"] = 0
    state["final_answer"] = None
    
    state["messages"].append({
        "role": "user",
        "content": f"""
根据当前执行结果，任务计划已经更新。

新的任务计划：

{new_plan}

请继续根据新的计划完成剩余任务。
不要重复已经完成的步骤。
"""
    })

    return state

def update_node(client, state):

    last_action = state["last_action"]

    if last_action["type"] == "final":

        state["final_answer"] = last_action["content"]

        update_plan_status(
            client,
            state,
            final_answer=last_action["content"]
        )

    else:

        update_plan_status(
            client,
            state
        )

    return state


def router(state):

    has_pending = any(
        item["status"] == "pending"
        for item in state["plan"]
    )

    action_type = state["last_action"]["type"]

    if action_type == "final":

        state["retry_count"] = 0

        if has_pending:
            return "replan"

        return "end"

    if action_type == "tool":

        has_failed_tool = any(
            not result["success"]
            for result in state["last_action"]["results"]
        )

        if has_failed_tool:

            if state["retry_count"] < MAX_RETRIES:
                return "retry"

            return "replan"

        state["retry_count"] = 0

    return "action"

def run_workflow(client, messages, user_input):

    state = {
        "messages": messages,
        "user_input": user_input,
        "plan": [],
        "tool_history": [],
        "last_action": None,
        "final_answer": None,
        "retry_count": 0
    }

    # 1. 制定计划
    state = planner_node(
        client,
        state
    )

    print("计划：")

    for item in state["plan"]:
        print(item)

    # 把用户任务和计划加入 LLM 上下文
    state["messages"].append({
        "role": "user",
        "content": f"""
用户任务：

{user_input}

任务计划：

{state["plan"]}

请参考计划完成任务。
如果实际工具结果与计划不一致，
可以根据实际结果调整执行方式。
"""
    })

    # 2. Workflow 主循环
    for _ in range(MAX_STEPS):

        # Action Node
        state = action_node(
            client,
            state
        )

        # Update Node
        state = update_node(
            client,
            state
        )

        print("当前状态：")

        for item in state["plan"]:
            print(item)

        route = router(state)

        if route == "end":
            return state

        elif route == "replan":

            state = replan_node(
                client,
                state
            )

        elif route == "retry":

            state = retry_node(
                state
            )

        elif route == "action":
            continue

    print("AI：达到最大执行轮数，任务停止。")

    return state