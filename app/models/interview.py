from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import IdentifierMixin, InterviewStatusEnum, TenantScopedMixin, TimestampMixin, enum_column


class Interview(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "interviews"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    interviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    round_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[InterviewStatusEnum] = enum_column(InterviewStatusEnum, InterviewStatusEnum.pending)
    scheduled_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scheduled_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)
    meeting_url: Mapped[str | None] = mapped_column(String(500))
    calendar_event_id: Mapped[str | None] = mapped_column(String(255), index=True)
    transcript: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    candidate = relationship("Candidate", back_populates="interviews")
    interviewer = relationship("User", back_populates="assigned_interviews")
    feedback_items = relationship("Feedback", back_populates="interview", cascade="all, delete-orphan")

