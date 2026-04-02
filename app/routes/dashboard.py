from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.dashboard import DashboardSummary
from app.services.analytics_service import AnalyticsService
from app.services.auth import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
analytics_service = AnalyticsService()


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(
    session: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
) -> DashboardSummary:
    return await analytics_service.get_summary(session, user.company_id)

