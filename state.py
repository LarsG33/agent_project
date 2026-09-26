import operator
from typing import Annotated, Any, Literal, TypedDict


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
    messages: Annotated[list[Any], operator.add]
    user_input: str
    plan: list[PlanStep]
    tool_history: Annotated[list[ToolHistoryRecord], operator.add]
    last_action: LastAction | None
    final_answer: str | None
    retry_count: int

    # 覆盖字段：每轮模型调用后递增，限制业务轮数。
    step_count: int
