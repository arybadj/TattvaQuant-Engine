from sqlalchemy import Boolean, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import IdentifierMixin, TenantScopedMixin, TimestampMixin


class Feedback(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "feedback"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    interview_id: Mapped[int | None] = mapped_column(ForeignKey("interviews.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    round_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    strengths: Mapped[dict] = mapped_column(JSONB, default=list, nullable=False)
    concerns: Mapped[dict] = mapped_column(JSONB, default=list, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    should_advance: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    candidate = relationship("Candidate", back_populates="feedback_items")
    interview = relationship("Interview", back_populates="feedback_items")

