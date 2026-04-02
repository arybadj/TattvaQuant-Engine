from collections.abc import Sequence

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.user import User
from app.utils.security import create_access_token, decode_token, verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User | None:
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def issue_token(user: User) -> str:
    return create_access_token(subject=str(user.id), company_id=user.company_id, role=user.role.value)


async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
        company_id = int(payload.get("company_id"))
    except (JWTError, ValueError, TypeError):
        raise credentials_exception

    user = (
        await session.execute(select(User).where(User.id == user_id, User.company_id == company_id, User.is_active))
    ).scalar_one_or_none()
    if not user:
        raise credentials_exception
    return user


def require_roles(*roles: Sequence[str] | str):
    flattened = {item for role in roles for item in (role if isinstance(role, (list, tuple, set)) else [role])}

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role.value not in flattened:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency
