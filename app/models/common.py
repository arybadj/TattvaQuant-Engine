import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column


class RoleEnum(str, enum.Enum):
    admin = "admin"
    hr = "hr"
    interviewer = "interviewer"


class CandidateStatusEnum(str, enum.Enum):
    uploaded = "uploaded"
    under_review = "under_review"
    approved = "approved"
    rejected = "rejected"
    email_sent = "email_sent"
    call_pending = "call_pending"
    scheduled = "scheduled"
    interviewing = "interviewing"
    completed = "completed"
    failed = "failed"


class DecisionEnum(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    hold = "hold"


class InterviewStatusEnum(str, enum.Enum):
    pending = "pending"
    scheduled = "scheduled"
    completed = "completed"
    cancelled = "cancelled"
    no_show = "no_show"


class WorkflowStatusEnum(str, enum.Enum):
    pending = "pending"
    retrying = "retrying"
    failed = "failed"
    completed = "completed"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TenantScopedMixin:
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), index=True, nullable=False)


class IdentifierMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)


def enum_column(enum_cls: type[enum.Enum], default: enum.Enum | None = None):
    return mapped_column(Enum(enum_cls, native_enum=False), default=default, nullable=False)


def short_string(length: int = 255) -> Mapped[str]:
    return mapped_column(String(length), nullable=False)

