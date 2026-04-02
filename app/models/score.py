from sqlalchemy import Float, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import IdentifierMixin, TenantScopedMixin, TimestampMixin


class Score(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "scores"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True, nullable=False)
    ats_score: Mapped[float] = mapped_column(Float, nullable=False)
    llm_score: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_score: Mapped[float] = mapped_column(Float, nullable=False)
    match_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    experience_summary: Mapped[str | None] = mapped_column(Text)

    candidate = relationship("Candidate", back_populates="scores")

