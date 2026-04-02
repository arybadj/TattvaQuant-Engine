from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import CandidateStatusEnum, IdentifierMixin, TenantScopedMixin, TimestampMixin, enum_column


class Candidate(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "candidates"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    phone: Mapped[str | None] = mapped_column(String(50))
    current_title: Mapped[str | None] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[CandidateStatusEnum] = enum_column(CandidateStatusEnum, CandidateStatusEnum.uploaded)
    final_score: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    company = relationship("Company", back_populates="candidates")
    job = relationship("Job", back_populates="candidates")
    resume = relationship("Resume", back_populates="candidate", uselist=False, cascade="all, delete-orphan")
    scores = relationship("Score", back_populates="candidate", cascade="all, delete-orphan")
    decisions = relationship("HRDecision", back_populates="candidate", cascade="all, delete-orphan")
    interviews = relationship("Interview", back_populates="candidate", cascade="all, delete-orphan")
    feedback_items = relationship("Feedback", back_populates="candidate", cascade="all, delete-orphan")
    workflow_states = relationship("WorkflowState", back_populates="candidate", cascade="all, delete-orphan")

