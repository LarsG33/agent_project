"""离线验证真实 LangGraph 调度；不访问 DeepSeek、不下载 embedding 模型。"""
import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

# 仅替代 RAG 的外部模型依赖；工具执行、Reducer 和图引擎仍用真实代码。
rag_stub = ModuleType("rag")
rag_stub.retrieve = lambda query: []
with patch.dict(sys.modules, {"rag": rag_stub}):
    import workflow
    import agent


def plan(status="pending"):
    return [{"step": 1, "task": "完成任务", "success_condition": "得到结果", "status": status}]


def tool(name="calculator", args='{"a": 10, "b": 20}', call_id="call_1"):
    return {"role": "assistant", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": args},
    }]}


def final(content="30"):
    return {"role": "assistant", "content": content}


class WorkflowTests(unittest.TestCase):
    def run_case(self, responses, updater=None):
        history = [{"role": "system", "content": "test"}]
        original = copy.deepcopy(history)
        def default_update(client, state, final_answer=None):
            return plan("done" if final_answer is not None else "pending")
        with patch.object(workflow, "retrieve_memory", return_value={}), \
             patch.object(workflow, "make_plan", return_value=plan()), \
             patch.object(workflow, "call_agent", side_effect=responses), \
             patch.object(workflow, "update_plan_status", side_effect=updater or default_update), \
             patch.object(workflow, "replan", return_value=plan()) as replanner, \
             redirect_stdout(io.StringIO()) as output:
            result = workflow.run_workflow(None, history, "计算10+20")
        self.assertEqual(history, original)
        return result, replanner.call_count, output.getvalue()

    def test_tool_loop_and_message_reducer(self):
        result, _, log = self.run_case([tool(), final()])
        self.assertEqual(result["final_answer"], "30")
        self.assertEqual(len(result["messages"]), 5)  # system/user/call/tool/final
        self.assertEqual(len(result["tool_history"]), 1)
        self.assertEqual(len(result["plan"]), 1)  # 覆盖，而非追加旧计划
        self.assertEqual(result["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(result["step_count"], 2)
        self.assertIn("[Node] tools", log)

    def test_direct_final_without_tools(self):
        result, _, log = self.run_case([final("你好")])
        self.assertEqual(result["final_answer"], "你好")
        self.assertEqual(result["tool_history"], [])
        self.assertNotIn("[Node] tools", log)

    def test_failure_retry_then_success(self):
        result, replans, log = self.run_case([tool(args="bad JSON"), tool(call_id="call_2"), final()])
        self.assertEqual([x["success"] for x in result["tool_history"]], [False, True])
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(replans, 0)
        self.assertIn("[Node] retry", log)

    def test_retry_exhaustion_replans(self):
        result, replans, _ = self.run_case([tool("missing") for _ in range(3)] + [final("无法完成")])
        self.assertEqual(replans, 1)
        self.assertEqual(len(result["tool_history"]), 3)

    def test_pending_final_replans(self):
        def updater(client, state, final_answer=None):
            return plan("done" if state["step_count"] == 2 else "pending")
        result, replans, _ = self.run_case([final("候选"), final("完成")], updater)
        self.assertEqual(replans, 1)
        self.assertEqual(result["final_answer"], "完成")

    def test_limit_stops_without_saving_candidate(self):
        with patch.object(workflow, "MAX_STEPS", 2):
            result, _, log = self.run_case([final("未完成"), final("仍未完成")], lambda *a, **kw: plan())
        self.assertEqual(result["step_count"], 2)
        self.assertIsNone(result["final_answer"])
        self.assertIn("[Node] stop", log)

    def test_multiple_tool_calls_and_errors_keep_ids(self):
        message = tool()
        message["tool_calls"] += tool("missing", call_id="call_2")["tool_calls"]
        message["tool_calls"] += tool(args="[]", call_id="call_3")["tool_calls"]
        with redirect_stdout(io.StringIO()):
            update = agent.execute_tools(message)
        self.assertEqual([m["tool_call_id"] for m in update["messages"]], ["call_1", "call_2", "call_3"])
        self.assertEqual([r["success"] for r in update["tool_history"]], [True, False, False])

    def test_model_call_uses_latest_plan_without_mutating_state(self):
        state = {"messages": [{"role": "user", "content": "test"}], "plan": plan()}
        original = copy.deepcopy(state)
        captured = {}
        class Message:
            def model_dump(self, **kwargs):
                return final()
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=Message())])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(agent.call_agent(client, state), final())
        self.assertEqual(state, original)
        self.assertEqual(len(captured["messages"]), 2)
        self.assertIn("当前最新任务计划", captured["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
