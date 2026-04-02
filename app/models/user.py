from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base
from app.models.common import IdentifierMixin, RoleEnum, TenantScopedMixin, TimestampMixin, enum_column


class User(IdentifierMixin, TenantScopedMixin, TimestampMixin, Base):
    __tablename__ = "users"

    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[RoleEnum] = enum_column(RoleEnum, RoleEnum.hr)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC", nullable=False)

    company = relationship("Company", back_populates="users")
    assigned_interviews = relationship("Interview", back_populates="interviewer")

