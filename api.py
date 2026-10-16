"""Day 20: minimal local HTTP interface for the existing LangGraph workflow.

Run only on 127.0.0.1. There is no authentication or user isolation.
"""
import os
import sqlite3
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from main_config import build_system_prompt
from task_manager import init_db, get_failed_tasks, get_pending_tasks
from workflow import run_workflow

load_dotenv()
init_db()
app = FastAPI(title="Agent Project API", version="1.0")


def make_client():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY is not configured")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com",
                  max_retries=2, timeout=30.0)


def messages():
    # Each HTTP request uses a fresh conversation context. Task state itself
    # is restored from LangGraph's checkpoint, not from this message list.
    return [{"role": "system", "content": build_system_prompt()}]


def task_status(task_id: str):
    with sqlite3.connect("tasks.sqlite") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE thread_id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return row[0]


class NewTask(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)


class Decision(BaseModel):
    approved: bool


class RetryRequest(BaseModel):
    # Explicit acknowledgement: retries can repeat non-idempotent actions.
    confirm_safe_to_retry: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    return {
        "paused": [{"id": tid, "title": title} for tid, title in get_pending_tasks()],
        "failed": [{"id": tid, "title": title} for tid, title in get_failed_tasks()],
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    return {"id": task_id, "status": task_status(task_id)}


@app.post("/tasks")
def create_task(payload: NewTask):
    task_id = "task-" + uuid4().hex
    state = run_workflow(make_client(), messages(), user_input=payload.prompt,
                         new_thread_id=task_id, interactive=False)
    return {"id": task_id, "status": task_status(task_id),
            "answer": state.get("final_answer") if state else None}


@app.post("/tasks/{task_id}/decision")
def decide(task_id: str, payload: Decision):
    if task_status(task_id) != "paused":
        raise HTTPException(status_code=409, detail="Task is not pending approval")
    state = run_workflow(make_client(), messages(), thread_id=task_id,
                         approval=payload.approved, interactive=False)
    return {"id": task_id, "status": task_status(task_id),
            "answer": state.get("final_answer") if state else None}


@app.post("/tasks/{task_id}/retry")
def retry(task_id: str, payload: RetryRequest):
    if task_status(task_id) != "failed":
        raise HTTPException(status_code=409, detail="Task is not failed")
    if not payload.confirm_safe_to_retry:
        raise HTTPException(status_code=400, detail="Explicit retry confirmation required")
    state = run_workflow(make_client(), messages(), thread_id=task_id,
                         interactive=False, retry_confirmed=True)
    return {"id": task_id, "status": task_status(task_id),
            "answer": state.get("final_answer") if state else None}
