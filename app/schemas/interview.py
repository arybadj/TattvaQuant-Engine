from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    recommendation: str
    should_advance: bool = False


class ScheduleRequest(BaseModel):
    candidate_id: int
    preferred_slots: list[datetime] = Field(default_factory=list)
    timezone: str

