from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from app.models.common import CandidateStatusEnum, DecisionEnum, InterviewStatusEnum
from app.schemas.common import ORMModel, TimestampedResponse


class CandidateUploadResponse(BaseModel):
    candidate_id: int
    workflow_state_id: int
    status: str


class ScoreSummary(ORMModel):
    ats_score: float
    llm_score: float
    embedding_score: float
    match_percentage: float
    final_score: float
    reasoning: dict[str, Any]
    experience_summary: str | None = None


class ResumeResponse(ORMModel):
    filename: str
    content_type: str
    raw_text: str
    structured_data: dict[str, Any]


class DecisionCreate(BaseModel):
    decision: DecisionEnum
    notes: str | None = None


class DecisionResponse(ORMModel):
    id: int
    candidate_id: int
    reviewer_id: int | None = None
    decision: DecisionEnum
    notes: str | None = None
    created_at: datetime


class InterviewResponse(ORMModel):
    id: int
    round_number: int
    status: InterviewStatusEnum
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str
    meeting_url: str | None = None


class CandidateDetailResponse(TimestampedResponse):
    company_id: int
    job_id: int
    full_name: str
    email: EmailStr
    phone: str | None = None
    current_title: str | None = None
    location: str | None = None
    status: CandidateStatusEnum
    final_score: float | None = None
    notes: str | None = None
    resume: ResumeResponse | None = None
    latest_score: ScoreSummary | None = None
    latest_decision: DecisionResponse | None = None
    interviews: list[InterviewResponse] = Field(default_factory=list)

