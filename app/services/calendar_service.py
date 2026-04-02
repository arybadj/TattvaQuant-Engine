from datetime import datetime, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.common import InterviewStatusEnum, RoleEnum
from app.models.interview import Interview
from app.models.user import User
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class CalendarService:
    def _google_client(self):
        if not settings.google_service_account_file:
            return None
        credentials = service_account.Credentials.from_service_account_file(
            settings.google_service_account_file,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)

    async def allocate_interviewer(self, session: AsyncSession, company_id: int) -> User | None:
        users = (
            await session.execute(
                select(User)
                .where(User.company_id == company_id, User.role == RoleEnum.interviewer, User.is_active)
                .order_by(User.id)
            )
        ).scalars().all()
        if not users:
            return None

        best_user = None
        best_count = None
        for user in users:
            current_count = (
                await session.execute(
                    select(func.count(Interview.id)).where(
                        Interview.company_id == company_id,
                        Interview.interviewer_id == user.id,
                        Interview.status.in_([InterviewStatusEnum.pending, InterviewStatusEnum.scheduled]),
                    )
                )
            ).scalar_one()
            if best_count is None or current_count < best_count:
                best_user = user
                best_count = current_count
        return best_user

    async def detect_conflicts(
        self,
        session: AsyncSession,
        company_id: int,
        interviewer_id: int,
        slot_start: datetime,
        slot_end: datetime,
    ) -> bool:
        clash = (
            await session.execute(
                select(Interview).where(
                    Interview.company_id == company_id,
                    Interview.interviewer_id == interviewer_id,
                    Interview.scheduled_start < slot_end,
                    Interview.scheduled_end > slot_start,
                )
            )
        ).scalar_one_or_none()
        return clash is not None

    async def book_slot(
        self,
        *,
        candidate_id: int,
        company_id: int,
        interviewer_id: int | None,
        round_number: int,
        timezone_name: str,
        session: AsyncSession,
    ) -> Interview:
        slot_start = datetime.now(timezone.utc) + timedelta(days=2)
        slot_end = slot_start + timedelta(minutes=45)
        interview = Interview(
            company_id=company_id,
            candidate_id=candidate_id,
            interviewer_id=interviewer_id,
            round_number=round_number,
            status=InterviewStatusEnum.scheduled,
            scheduled_start=slot_start,
            scheduled_end=slot_end,
            timezone=timezone_name,
            meeting_url=f"https://meet.example.com/interviews/{candidate_id}-{round_number}",
            metadata_json={"source": "calendar_service"},
        )
        session.add(interview)
        await session.flush()

        google = self._google_client()
        if google:
            try:
                event = {
                    "summary": f"Candidate interview #{candidate_id}",
                    "start": {"dateTime": slot_start.isoformat(), "timeZone": timezone_name},
                    "end": {"dateTime": slot_end.isoformat(), "timeZone": timezone_name},
                }
                response = google.events().insert(calendarId="primary", body=event).execute()
                interview.calendar_event_id = response.get("id")
            except Exception as exc:
                logger.exception("google_calendar_booking_failed", error=str(exc), candidate_id=candidate_id)

        logger.info("calendar_slot_booked", candidate_id=candidate_id, interview_id=interview.id)
        return interview
