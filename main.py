from openai import OpenAI
from dotenv import load_dotenv
from workflow import run_workflow

import os


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def build_system_prompt():

    return """
你是一名Python老师。

回答要简洁，并尽量使用简单例子解释。

你是任务执行 Agent，
可以根据用户任务和任务计划调用工具。

涉及项目内部事实的问题，
必须优先调用 search_knowledge。

不要编造项目内部事实。


你可以使用 save_user_memory 工具保存长期记忆。

只有满足以下条件的信息才应该保存：

1. 信息比较稳定，不是一次性的。
2. 未来可能再次使用。
3. 这条信息可能影响未来回答或任务执行。

适合保存的信息包括：

- 用户长期偏好
- 用户长期目标
- 项目的固定配置
- 用户明确要求记住的信息

不要保存：

- 一次性问题
- 临时状态
- 单次计算结果
- 普通聊天内容
- 当前 workflow 的中间结果

如果用户要求删除、忘记或者纠正某条长期记忆，
可以使用 delete_user_memory。
"""


messages = [
    {
        "role": "system",
        "content": build_system_prompt()
    }
]


while True:

    user_input = input("你：")

    if user_input == "exit":
        break

    elif user_input == "clear":
        messages = messages[:1]
        continue

    state = run_workflow(
        client,
        messages,
        user_input
    )

    # 只把真正的用户输入和最终回答
    # 保存到正式聊天历史
    messages.append({
        "role": "user",
        "content": user_input
    })

    if state["final_answer"]:
        messages.append({
            "role": "assistant",
            "content": state["final_answer"]
        })
