from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import IdentifierMixin, TenantScopedMixin, TimestampMixin, WorkflowStatusEnum, enum_column


class WorkflowState(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "workflow_states"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    interview_round: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    current_node: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[WorkflowStatusEnum] = enum_column(WorkflowStatusEnum, WorkflowStatusEnum.pending)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_action: Mapped[str | None] = mapped_column(String(255))
    state_payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    candidate = relationship("Candidate", back_populates="workflow_states")

