from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candidate import Candidate
from app.models.common import CandidateStatusEnum
from app.schemas.dashboard import AnalyticsOverview, DashboardSummary, FunnelPoint


class AnalyticsService:
    async def get_summary(self, session: AsyncSession, company_id: int) -> DashboardSummary:
        total = await self._count(session, company_id)
        shortlisted = await self._count(session, company_id, Candidate.status == CandidateStatusEnum.approved)
        scheduled = await self._count(session, company_id, Candidate.status == CandidateStatusEnum.scheduled)
        completed = await self._count(session, company_id, Candidate.status == CandidateStatusEnum.completed)
        failed = await self._count(session, company_id, Candidate.status == CandidateStatusEnum.failed)

        return DashboardSummary(
            total_candidates=total,
            shortlisted_count=shortlisted,
            scheduled_count=scheduled,
            completed_count=completed,
            shortlisted_percentage=round((shortlisted / total) * 100, 2) if total else 0.0,
            scheduled_percentage=round((scheduled / total) * 100, 2) if total else 0.0,
            conversion_rate=round((completed / total) * 100, 2) if total else 0.0,
            dropoff_count=failed,
        )

    async def overview(self, session: AsyncSession, company_id: int) -> AnalyticsOverview:
        summary = await self.get_summary(session, company_id)
        rows = (
            await session.execute(
                select(Candidate.status, func.count(Candidate.id)).where(Candidate.company_id == company_id).group_by(Candidate.status)
            )
        ).all()
        by_status = {status.value: count for status, count in rows}
        funnel = [
            FunnelPoint(label="Uploaded", value=summary.total_candidates),
            FunnelPoint(label="Shortlisted", value=summary.shortlisted_count),
            FunnelPoint(label="Scheduled", value=summary.scheduled_count),
            FunnelPoint(label="Completed", value=summary.completed_count),
        ]
        return AnalyticsOverview(summary=summary, funnel=funnel, by_status=by_status)

    async def _count(self, session: AsyncSession, company_id: int, *filters) -> int:
        return (
            await session.execute(select(func.count(Candidate.id)).where(Candidate.company_id == company_id, *filters))
        ).scalar_one()

