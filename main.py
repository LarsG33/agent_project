from openai import OpenAI
from dotenv import load_dotenv
import json
import os
def calculator(a, b):
    print("本地加法工具正在调用")
    return a + b

def multiply(a, b):
    print("本地乘法工具正在调用")
    return a * b
load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [
    {
        "role": "system",
        "content": "你是一名Python老师。回答要简洁，并尽量使用简单例子解释。"
    }
]

tools = [
    {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
            },
            "required": ["a", "b"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]
tool_map = {
    "calculator": calculator,
    "multiply": multiply
}
MAX_STEPS = 10

def run_agent():
    for step in range(MAX_STEPS):

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if not message.tool_calls:
            print("AI：", message.content)
            messages.append(message)
            return

        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            func = tool_map.get(function_name)

            if func is None:
                result = f"Error: tool '{function_name}' does not exist."
            else:
                try:
                    result = func(**arguments)
                except Exception as e:
                    result = f"Tool execution error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    print("AI：达到最大工具调用轮数，任务停止。")
    
while True:
    user_input = input("你：")
    if user_input == "exit":
        break
    elif user_input == "clear":
        messages = messages[:1]
        continue
    messages.append({
        "role": "user",
        "content": user_input
    })
    run_agent()





