import json


def build_context(
    messages,
    user_input: str,
    relevant_memory,
    plan
):
    # 正式聊天历史的副本
    # 后续修改不会影响原来的 messages
    agent_messages = messages.copy()

    if relevant_memory:
        memory_text = json.dumps(
            relevant_memory,
            ensure_ascii=False,
            indent=2
        )

        agent_messages.append({
            "role": "system",
            "content": f"""
以下是与当前任务相关的用户长期记忆：

{memory_text}

只在与当前任务有关时使用这些信息。
不要把这些记忆当作用户本轮新说的话。
"""
        })

    plan_text = json.dumps(
        plan,
        ensure_ascii=False,
        indent=2
    )

    agent_messages.append({
        "role": "user",
        "content": f"""
用户任务：

{user_input}

任务计划：

{plan_text}

请参考计划完成任务。

如果实际工具结果与计划不一致，
可以根据实际结果调整执行方式。

不要重复已经完成的步骤。
"""
    })

    return agent_messages
