from typing import Any, Literal, TypedDict


class WorkflowGraphState(TypedDict, total=False):
    candidate_id: int
    company_id: int
    status: Literal["pending", "retrying", "failed", "completed", "approved", "rejected"]
    stage: str
    interview_round: int
    retry_count: int
    last_action: str
    schedule: dict[str, Any]
    feedback: dict[str, Any]
    next_action: str
    error_message: str | None

