import json
from typing import cast

from state import AgentState, PlanStep


def make_plan(
    client,
    user_input: str,
    relevant_memory=None,
    conversation_history=None
) -> list[PlanStep]:

    planner_messages = [
        {
            "role": "system",
            "content": """
你是任务规划助手。

当前 Agent 可以使用：

1. calculator：两个数字相加
2. multiply：两个数字相乘
3. search_knowledge：查询项目知识库
4. save_user_memory：保存值得长期记住的信息
5. delete_user_memory：删除错误、失效或用户要求删除的长期记忆

制定计划时遵守以下规则：

1. 步骤必须按照真实执行顺序排列。
2. 每一步只做一个明确目标。
3. 不要创建重复或含义重叠的步骤。
4. 如果一个计算需要多个工具，按照实际运算顺序拆分。
5. 查询类任务如果知识库没有相关信息，该步骤可以最终标记为 failed。
6. 最后可以有一个汇总结果的步骤。
7. 只制定计划，不要执行任务。
8. 只有长期稳定且未来有用的信息才需要保存为长期记忆。
9. 不要把普通问题、临时状态或计算结果保存为长期记忆。
"""
        }
    ]

    if conversation_history:
        planner_messages.extend(
            conversation_history[1:]
        )

    if relevant_memory:
        memory_text = json.dumps(
            relevant_memory,
            ensure_ascii=False,
            indent=2
        )

        planner_messages.append({
            "role": "system",
            "content": f"""
以下是与当前任务相关的用户长期记忆：

{memory_text}

制定计划时可以使用这些信息，
不要重复查询已经知道的信息。
"""
        })

    prompt = f"""
用户任务是：

{user_input}

请把任务拆成几个简单步骤。

只输出 JSON 数组，不要输出其他内容。

格式必须类似：

[
  {{
    "step": 1,
    "task": "第一步要做什么",
    "success_condition": "怎样才算这一步完成",
    "status": "pending"
  }},
  {{
    "step": 2,
    "task": "第二步要做什么",
    "success_condition": "怎样才算这一步完成",
    "status": "pending"
  }}
]
"""

    planner_messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=planner_messages
    )

    plan_text = response.choices[0].message.content or "[]"

    plan = json.loads(plan_text)

    return cast(list[PlanStep], plan)


def update_plan_status(
    client,
    state: AgentState,
    final_answer: str | None = None
) -> None:

    prompt = f"""
当前计划：

{json.dumps(state["plan"], ensure_ascii=False)}

目前已经执行的工具记录：

{json.dumps(state["tool_history"], ensure_ascii=False)}

最终回答：

{final_answer if final_answer else "尚未生成最终回答"}

请根据：
1. 每个步骤的 task
2. success_condition
3. 实际工具调用和结果
4. 如果已经存在最终回答，也结合最终回答判断

更新每个步骤状态。

状态只能是：
pending
done
failed

注意：
- 工具执行成功，不代表任务步骤一定完成。
- 一个步骤可能需要多个工具调用。
- 有些步骤不需要工具，比如“汇总结果”，这类步骤可以根据最终回答判断是否完成。
- 不要假设一个工具调用对应一个计划步骤。

只返回完整 JSON 数组。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你负责根据实际执行情况更新任务计划状态。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    plan_text = response.choices[0].message.content or "[]"

    plan = json.loads(plan_text)

    state["plan"] = cast(list[PlanStep], plan)


def replan(
    client,
    state: AgentState,
    relevant_memory=None
) -> list[PlanStep]:

    replan_messages = [
        {
            "role": "system",
            "content": "你负责根据当前执行情况重新规划剩余任务。"
        }
    ]

    if relevant_memory:
        memory_text = json.dumps(
            relevant_memory,
            ensure_ascii=False,
            indent=2
        )

        replan_messages.append({
            "role": "system",
            "content": f"""
以下是与当前任务相关的用户长期记忆：

{memory_text}

重新规划时可以继续参考这些信息。
"""
        })

    prompt = f"""
原始用户任务：

{state["user_input"]}

当前计划：

{json.dumps(state["plan"], ensure_ascii=False)}

已经执行的工具记录：

{json.dumps(state["tool_history"], ensure_ascii=False)}

当前最终回答：

{state["final_answer"] if state["final_answer"] else "尚未生成"}

请根据实际执行情况重新调整任务计划。

要求：

1. 已经完成的步骤保留为 done。
2. 已经确认失败且无法完成的步骤可以保留为 failed。
3. 对仍未完成的任务重新规划必要步骤。
4. 不要重复已经完成的工作。
5. 新增步骤状态必须是 pending。
6. 只返回完整 JSON 数组。
"""

    replan_messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=replan_messages
    )

    plan_text = response.choices[0].message.content or "[]"

    plan = json.loads(plan_text)

    return cast(list[PlanStep], plan)
