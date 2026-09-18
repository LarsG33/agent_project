from typing import Any, Literal, TypedDict


class PlanStep(TypedDict):
    step: int
    task: str
    success_condition: str
    status: Literal["pending", "done", "failed"]


class ToolHistoryRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    result: str
    success: bool


class ToolStepResult(TypedDict):
    tool: str
    result: str
    success: bool


class ToolAction(TypedDict):
    type: Literal["tool"]
    results: list[ToolStepResult]


class FinalAction(TypedDict):
    type: Literal["final"]
    content: str


LastAction = ToolAction | FinalAction


class AgentState(TypedDict):
    messages: list[Any]
    user_input: str
    plan: list[PlanStep]
    tool_history: list[ToolHistoryRecord]
    last_action: LastAction | None
    final_answer: str | None
    retry_count: int