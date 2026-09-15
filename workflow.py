from planner import make_plan, update_plan_status
from agent import run_agent_step


MAX_STEPS = 10


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

    if state["last_action"]["type"] == "final":
        return "end"

    return "action"


def run_workflow(client, messages, user_input):

    state = {
        "messages": messages,
        "user_input": user_input,
        "plan": [],
        "tool_history": [],
        "last_action": None,
        "final_answer": None
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

        # Router
        route = router(state)

        if route == "end":
            return state

    print("AI：达到最大执行轮数，任务停止。")

    return state