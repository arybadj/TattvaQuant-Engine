from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from app.database.base import Base
from app.models.common import IdentifierMixin, TenantScopedMixin, TimestampMixin


class Embedding(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "embeddings"

    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)

    job = relationship("Job", back_populates="embeddings")

