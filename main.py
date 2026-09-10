from openai import OpenAI
from dotenv import load_dotenv
from rag import retrieve
import json
import os
def calculator(a, b):
    print("本地加法工具正在调用")
    return a + b

def multiply(a, b):
    print("本地乘法工具正在调用")
    return a * b
load_dotenv()
def search_knowledge(query):
    result = retrieve(query)

    if not result:
        return "No relevant information found in the knowledge base."

    return "\n".join(result)

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

对于涉及本项目内部事实的问题，必须优先调用 search_knowledge 工具，
不要凭模型自身知识猜测。

如果工具返回了知识库检索结果，应优先根据这些结果回答。
如果知识库中没有相关信息，不要编造项目相关事实。
"""
    }
]

tools = [
    {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Use this tool when the user asks to multiply two numeric values.",
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
            "description": "Use this tool when the user asks to add two numeric values.",
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
        "name": "search_knowledge",
        "description": "Use this tool when the user asks about private information specific to this project.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or keywords to search for."
                }
            },
            "required": ["query"]
        }
    }
}
]
tool_map = {
    "calculator": calculator,
    "multiply": multiply,
    "search_knowledge": search_knowledge
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

            print(f"[Tool] {function_name}")
            print(f"[Args] {arguments}")

            func = tool_map.get(function_name)

            if func is None:
                result = f"Error: tool '{function_name}' does not exist."
            else:
                try:
                    result = func(**arguments)
                except Exception as e:
                    result = f"Tool execution error: {e}"

            print(f"[Result] {result}")

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





