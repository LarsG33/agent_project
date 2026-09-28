from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from planner import make_plan, update_plan_status, replan
from agent import call_agent, execute_tools
from state import AgentState
from memory import retrieve_memory
from context import build_context


MAX_STEPS = 10  # 模型执行轮数，不是图中节点总数
MAX_RETRIES = 2

# Day15：这两个工具会修改长期记忆，因此先让用户确认。
APPROVAL_TOOLS = {"save_user_memory", "delete_user_memory"}


def route_agent(state: AgentState) -> str:
    # 这里收到的是 agent 返回值已经合并后的 State。
    message = state["messages"][-1]
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        tool_names = {
            item["function"]["name"]
            for item in tool_calls
        }

        if tool_names & APPROVAL_TOOLS:
            return "approval"

        return "tools"

    return "update_plan"


def route_after_approval(state: AgentState) -> str:
    if state["approval_granted"]:
        return "tools"
    return "cancel"


def route_after_update(state: AgentState) -> str:
    last_action = state["last_action"]
    has_pending = any(item["status"] == "pending" for item in state["plan"])

    # 有最终回答且没有待办才结束；failed 可以作为已确认失败汇总。
    if last_action and last_action["type"] == "final" and not has_pending:
        return "end"

    if state["step_count"] >= MAX_STEPS:
        return "stop"

    if last_action and last_action["type"] == "final":
        return "replan"

    if last_action and last_action["type"] == "tool":
        if any(not result["success"] for result in last_action["results"]):
            if state["retry_count"] < MAX_RETRIES:
                return "retry"
            return "replan"

    return "agent"


def retry_node(state: AgentState) -> dict:
    print("[Node] retry")
    count = state["retry_count"] + 1
    return {
        "retry_count": count,
        "messages": [{
            "role": "user",
            "content": (
                f"上一轮工具执行失败，这是第 {count} 次重试。"
                "请根据错误信息重新尝试，必要时调整参数。"
                "只重试失败部分，不要重复已经成功的工具操作。"
            ),
        }],
    }


def stop_node(state: AgentState) -> dict:
    print("[Node] stop：达到最大执行轮数，任务停止。")
    # 未通过计划检查的候选回答不能当成最终回答保存。
    return {"final_answer": None}


def build_graph(client, conversation_history, relevant_memory):
    """外层固定依赖，内层 Node 只接收 State、返回本轮更新。"""
    def planner_node(state: AgentState) -> dict:
        print("[Node] planner")
        plan = make_plan(
            client, state["user_input"], relevant_memory, conversation_history
        )
        print("计划：")
        for item in plan:
            print(item)
        return {"plan": plan}

    def agent_node(state: AgentState) -> dict:
        print(f"[Node] agent（第 {state['step_count'] + 1} 轮）")
        message = call_agent(client, state)
        if message.get("tool_calls"):
            action = None  # 此时只有调用请求，还没有工具结果
        else:
            action = {"type": "final", "content": message.get("content") or ""}
        return {
            "messages": [message],  # 只返回新增消息，Reducer 负责追加
            "last_action": action,
            "final_answer": None,
            "step_count": state["step_count"] + 1,
            "approval_granted": None,
        }

    def approval_node(state: AgentState) -> dict:
        print("[Node] approval")
        print("以下工具会修改长期记忆，需要用户确认：")

        message = state["messages"][-1]
        for tool_call in message.get("tool_calls", []):
            name = tool_call["function"]["name"]
            if name in APPROVAL_TOOLS:
                print(f"- {name}: {tool_call['function']['arguments']}")

        # 第一次运行到这里会暂停。
        # Command(resume=...) 恢复后，这个 Node 会从头重新执行，
        # 再次运行到 interrupt() 时，approved 会得到 resume 的值。
        approved = interrupt("是否允许执行以上长期记忆修改？")

        return {
            "approval_granted": bool(approved)
        }

    def cancel_node(state: AgentState) -> dict:
        print("[Node] cancel：用户拒绝了工具执行。")
        answer = "用户拒绝了长期记忆修改，本次操作已取消。"
        return {
            "last_action": {
                "type": "final",
                "content": answer,
            },
            "final_answer": answer,
        }

    def tools_node(state: AgentState) -> dict:
        print("[Node] tools")
        return execute_tools(state["messages"][-1])

    def update_plan_node(state: AgentState) -> dict:
        print("[Node] update_plan")
        action = state["last_action"]
        answer = action["content"] if action and action["type"] == "final" else None
        plan = update_plan_status(client, state, final_answer=answer)
        failed = bool(action and action["type"] == "tool" and any(
            not item["success"] for item in action["results"]
        ))
        print("当前状态：")
        for item in plan:
            print(item)
        return {
            "plan": plan,  # 没有追加 Reducer，整份替换
            "final_answer": answer,
            "retry_count": state["retry_count"] if failed else 0,
        }

    def replan_node(state: AgentState) -> dict:
        print("[Node] replan")
        return {
            "plan": replan(client, state, relevant_memory),
            "retry_count": 0,
            "final_answer": None,
        }

    # 阅读重点：原先 for / if 的调度现在由以下图结构负责。
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("approval", approval_node)
    graph.add_node("cancel", cancel_node)
    graph.add_node("tools", tools_node)
    graph.add_node("update_plan", update_plan_node)
    graph.add_node("retry", retry_node)
    graph.add_node("replan", replan_node)
    graph.add_node("stop", stop_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", route_agent, {
        "approval": "approval",
        "tools": "tools",
        "update_plan": "update_plan",
    })
    graph.add_conditional_edges("approval", route_after_approval, {
        "tools": "tools",
        "cancel": "cancel",
    })
    graph.add_edge("cancel", END)
    graph.add_edge("tools", "update_plan")
    graph.add_conditional_edges("update_plan", route_after_update, {
        "agent": "agent",
        "retry": "retry",
        "replan": "replan",
        "stop": "stop",
        "end": END,
    })
    graph.add_edge("retry", "agent")
    graph.add_edge("replan", "agent")
    graph.add_edge("stop", END)

    # interrupt() 必须配合 checkpointer。
    # Day15 先用内存版，只为观察暂停/恢复机制。
    return graph.compile(
        checkpointer=InMemorySaver()
    )


def run_workflow(client, messages, user_input: str) -> AgentState:
    relevant_memory = retrieve_memory(client, user_input)
    if relevant_memory:
        print("相关长期记忆：", relevant_memory)

    initial_state: AgentState = {
        "messages": build_context(messages, user_input, relevant_memory),
        "user_input": user_input,
        "plan": [],
        "tool_history": [],
        "last_action": None,
        "final_answer": None,
        "retry_count": 0,
        "step_count": 0,
        "approval_granted": None,
    }

    app = build_graph(client, messages, relevant_memory)

    # thread_id 相当于这一次任务的“存档槽编号”。
    # 恢复时必须继续使用同一个 config。
    config = {
        "recursion_limit": MAX_STEPS * 4 + 10,
        "configurable": {
            "thread_id": f"task-{uuid4().hex}"
        },
    }

    state = app.invoke(
        initial_state,
        config=config
    )

    # 如果图碰到 interrupt()，invoke 会先返回。
    # CLI 在图外询问用户，再用 Command(resume=...) 恢复同一 thread。
    while "__interrupt__" in state:
        answer = input("是否允许执行以上操作？(y/n)：").strip().lower()
        approved = answer in {"y", "yes", "1", "true", "允许", "同意"}

        state = app.invoke(
            Command(resume=approved),
            config=config
        )

    if state["final_answer"]:
        print("AI：", state["final_answer"])

    return state
