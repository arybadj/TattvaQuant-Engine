from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_candidates: int
    shortlisted_count: int
    scheduled_count: int
    completed_count: int
    shortlisted_percentage: float
    scheduled_percentage: float
    conversion_rate: float
    dropoff_count: int


class FunnelPoint(BaseModel):
    label: str
    value: int


class AnalyticsOverview(BaseModel):
    summary: DashboardSummary
    funnel: list[FunnelPoint]
    by_status: dict[str, int]

