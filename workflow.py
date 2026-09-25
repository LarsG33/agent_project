from planner import make_plan, update_plan_status, replan
from agent import run_agent_step
from state import AgentState
from memory import retrieve_memory
from context import build_context


MAX_STEPS = 10
MAX_RETRIES = 2


def retry_node(state: AgentState) -> AgentState:

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


def planner_node(
    client,
    state: AgentState,
    relevant_memory,
    messages
) -> AgentState:

    plan = make_plan(
        client,
        state["user_input"],
        relevant_memory,
        messages
    )

    state["plan"] = plan

    return state


def action_node(client, state: AgentState) -> AgentState:

    result = run_agent_step(
        client,
        state
    )

    state["last_action"] = result

    return state


def replan_node(
    client,
    state: AgentState,
    relevant_memory
) -> AgentState:

    new_plan = replan(
        client,
        state,
        relevant_memory
    )

    state["plan"] = new_plan
    state["retry_count"] = 0
    state["final_answer"] = None

    return state


def update_node(client, state: AgentState) -> AgentState:

    last_action = state["last_action"]

    if last_action is None:
        return state

    # LLM 已经给出最终回答
    if last_action["type"] == "final":

        state["final_answer"] = last_action["content"]

        update_plan_status(
            client,
            state,
            final_answer=last_action["content"]
        )

        state["retry_count"] = 0

    # 本轮执行了工具
    else:

        update_plan_status(
            client,
            state
        )

        has_failed_tool = any(
            not result["success"]
            for result in last_action["results"]
        )

        if not has_failed_tool:
            state["retry_count"] = 0

    return state


def router(state: AgentState) -> str:

    last_action = state["last_action"]

    if last_action is None:
        return "action"

    has_pending = any(
        item["status"] == "pending"
        for item in state["plan"]
    )

    if last_action["type"] == "final":

        if has_pending:
            return "replan"

        return "end"

    if last_action["type"] == "tool":

        has_failed_tool = any(
            not result["success"]
            for result in last_action["results"]
        )

        if has_failed_tool:

            if state["retry_count"] < MAX_RETRIES:
                return "retry"

            return "replan"

    return "action"


def run_workflow(
    client,
    messages,
    user_input: str
) -> AgentState:

    # 1. 检索与当前任务相关的长期记忆
    relevant_memory = retrieve_memory(
        client,
        user_input
    )

    if relevant_memory:
        print(
            "相关长期记忆：",
            relevant_memory
        )

    # 2. 创建初始 State
    state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "plan": [],
        "tool_history": [],
        "last_action": None,
        "final_answer": None,
        "retry_count": 0
    }

    # 3. 制定计划
    state = planner_node(
        client,
        state,
        relevant_memory,
        messages
    )

    print("计划：")

    for item in state["plan"]:
        print(item)

    # 4. 构造本轮临时 Agent 上下文
    agent_messages = build_context(
        messages,
        user_input,
        relevant_memory
    )

    state["messages"] = agent_messages

    # 5. Workflow 主循环
    for _ in range(MAX_STEPS):

        state = action_node(
            client,
            state
        )

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
                state,
                relevant_memory
            )

        elif route == "retry":

            state = retry_node(
                state
            )

        elif route == "action":
            continue

    print("AI：达到最大执行轮数，任务停止。")

    return state
