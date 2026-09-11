from rag import retrieve


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
    }
]


tool_map = {
    "calculator": calculator,
    "multiply": multiply,
    "search_knowledge": search_knowledge
}