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

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "multiply":
            result = multiply(arguments["a"], arguments["b"])

        elif function_name == "calculator":
            result = calculator(arguments["a"], arguments["b"])

            # 1. 把模型刚才的 tool call 记录下来
        messages.append(message)

        # 2. 把工具执行结果告诉模型
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

        # 3. 再调用一次模型
        response2 = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        answer = response2.choices[0].message.content

        print("AI：", answer)

        # 4. 保存最终回答
        messages.append({
            "role": "assistant",
            "content": answer
        })
    else:
        print("content:", message.content)

        messages.append({
            "role": "assistant",
            "content": message.content
        })
