from rag import retrieve
from memory import add_memory, delete_memory


def calculator(a, b):
    print("本地加法工具正在调用")
    return a + b


def multiply(a, b):
    print("本地乘法工具正在调用")
    return a * b


def search_knowledge(query):
    result = retrieve(query)

    if not result:
        return "No relevant information found in the knowledge base."

    return "\n".join(result)


def save_user_memory(key, value):
    add_memory(key, value)

    return f"已保存长期记忆：{key} = {value}"


def delete_user_memory(key):
    success = delete_memory(key)

    if success:
        return f"已删除长期记忆：{key}"

    return f"未找到长期记忆：{key}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Use this tool when the user asks to multiply two numeric values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
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
                    "a": {"type": "number"},
                    "b": {"type": "number"}
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
    },

    {
        "type": "function",
        "function": {
            "name": "save_user_memory",
            "description": (
                "Save information that is stable, useful in future conversations, "
                "and worth remembering long term, such as user preferences, "
                "long-term goals, or fixed project settings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "A short name describing the memory, "
                            "for example preferred_language."
                        )
                    },
                    "value": {
                        "type": "string",
                        "description": "The information that should be remembered."
                    }
                },
                "required": ["key", "value"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "delete_user_memory",
            "description": (
                "Delete long-term memory that is incorrect, outdated, "
                "or that the user asks to remove."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the memory that should be deleted."
                    }
                },
                "required": ["key"]
            }
        }
    }
]


tool_map = {
    "calculator": calculator,
    "multiply": multiply,
    "search_knowledge": search_knowledge,
    "save_user_memory": save_user_memory,
    "delete_user_memory": delete_user_memory
}