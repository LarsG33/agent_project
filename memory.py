import json
import os


MEMORY_FILE = "memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            memory,
            f,
            ensure_ascii=False,
            indent=4
        )


def add_memory(key, value):
    memory = load_memory()

    memory[key] = value

    save_memory(memory)


def delete_memory(key):
    memory = load_memory()

    if key in memory:
        del memory[key]
        save_memory(memory)
        return True

    return False


def retrieve_memory(
    client,
    user_input: str
):
    memory = load_memory()

    # 没有长期记忆时，
    # 不需要浪费一次 LLM 调用
    if not memory:
        return {}

    memory_text = json.dumps(
        memory,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
当前用户输入：

{user_input}

已有长期记忆：

{memory_text}

请从已有长期记忆中选择
与当前用户输入真正相关的内容。

要求：

1. 只能选择已有记忆。
2. 不要创造新的信息。
3. 不要修改原来的 key 和 value。
4. 只选择当前任务真正需要的信息。
5. 如果没有相关记忆，返回空字典 {{}}。
6. 只返回 JSON 对象，不要解释。

例如：

已有记忆：

{{
    "location": "Singapore",
    "language": "Python"
}}

用户输入：

我喜欢使用什么编程语言？

返回：

{{
    "language": "Python"
}}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你负责从用户长期记忆中，"
                    "检索与当前任务相关的信息。"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = (
        response.choices[0].message.content
        or "{}"
    ).strip()

    # 处理模型偶尔返回 Markdown JSON 代码块
    if answer.startswith("```"):
        answer = (
            answer
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:
        result = json.loads(answer)

    except json.JSONDecodeError:
        return {}

    if not isinstance(result, dict):
        return {}

    return result
