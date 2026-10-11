"""任务目录只存元数据；执行现场由 LangGraph Checkpointer 管理。"""
import sqlite3

DB_PATH = "tasks.sqlite"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)


def save_task(thread_id: str, title: str, status: str = "running"):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tasks (thread_id, title, status)
            VALUES (?, ?, ?)
        """, (thread_id, title, status))


def get_pending_tasks():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT thread_id, title FROM tasks
            WHERE status = 'paused'
            ORDER BY rowid
        """).fetchall()


def update_task_status(thread_id: str, status: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE tasks SET status = ? WHERE thread_id = ?
        """, (status, thread_id))
