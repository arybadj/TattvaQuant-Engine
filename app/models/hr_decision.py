from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import DecisionEnum, IdentifierMixin, TenantScopedMixin, TimestampMixin, enum_column


class HRDecision(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "hr_decisions"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision: Mapped[DecisionEnum] = enum_column(DecisionEnum, DecisionEnum.pending)
    notes: Mapped[str | None] = mapped_column(Text)

    candidate = relationship("Candidate", back_populates="decisions")

