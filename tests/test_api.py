"""Offline HTTP contract checks: model/workflow are mocked."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_empty_input_rejected(self):
        response = self.client.post("/tasks", json={"prompt": ""})
        self.assertEqual(response.status_code, 422)

    def test_retry_requires_confirmation(self):
        with patch("api.task_status", return_value="failed"):
            response = self.client.post(
                "/tasks/task-example/retry",
                json={"confirm_safe_to_retry": False},
            )
        self.assertEqual(response.status_code, 400)

    def test_wrong_status_rejects_approval(self):
        with patch("api.task_status", return_value="completed"):
            response = self.client.post(
                "/tasks/task-example/decision", json={"approved": True}
            )
        self.assertEqual(response.status_code, 409)

    def test_create_returns_task_id_and_answer(self):
        fake_state = {"final_answer": "30"}
        with patch("api.make_client", return_value=object()), \
             patch("api.run_workflow", return_value=fake_state) as run, \
             patch("api.task_status", return_value="completed"):
            response = self.client.post("/tasks", json={"prompt": "10+20"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "30")
        self.assertTrue(response.json()["id"].startswith("task-"))
        self.assertFalse(run.call_args.kwargs["interactive"])

    def test_approval_resume_uses_boolean(self):
        with patch("api.task_status", return_value="paused"), \
             patch("api.make_client", return_value=object()), \
             patch("api.run_workflow", return_value={"final_answer": "已取消"}) as run:
            response = self.client.post(
                "/tasks/task-example/decision", json={"approved": False}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIs(run.call_args.kwargs["approval"], False)


if __name__ == "__main__":
    unittest.main()
