from openai import OpenAI
from dotenv import load_dotenv
from workflow import run_workflow

import os


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


messages = [
    {
        "role": "system",
        "content": """
你是一名Python老师。

回答要简洁，并尽量使用简单例子解释。

你是任务执行 Agent，
可以根据用户任务和任务计划调用工具。

涉及项目内部事实的问题，
必须优先调用 search_knowledge。

不要编造项目内部事实。
"""
    }
]


while True:

    user_input = input("你：")

    if user_input == "exit":
        break

    elif user_input == "clear":
        messages = messages[:1]
        continue

    run_workflow(
        client,
        messages,
        user_input
    )