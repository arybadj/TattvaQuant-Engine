from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.schemas.dashboard import AnalyticsOverview
from app.services.analytics_service import AnalyticsService
from app.services.auth import get_current_user

router = APIRouter(prefix="/analytics", tags=["analytics"])
analytics_service = AnalyticsService()


@router.get("/overview", response_model=AnalyticsOverview)
async def analytics_overview(
    session: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
) -> AnalyticsOverview:
    return await analytics_service.overview(session, user.company_id)

