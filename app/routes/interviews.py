from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.feedback import Feedback
from app.models.interview import Interview
from app.schemas.interview import FeedbackCreate
from app.services.auth import get_current_user, require_roles
from app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/interviews", tags=["interviews"])
workflow_service = WorkflowService()


@router.get("/status")
async def interview_status(session: AsyncSession = Depends(get_db), user=Depends(get_current_user)) -> list[dict]:
    rows = (
        await session.execute(
            select(Interview).where(Interview.company_id == user.company_id).order_by(Interview.created_at.desc())
        )
    ).scalars()
    return [
        {
            "id": interview.id,
            "candidate_id": interview.candidate_id,
            "round_number": interview.round_number,
            "status": interview.status,
            "scheduled_start": interview.scheduled_start,
            "scheduled_end": interview.scheduled_end,
            "meeting_url": interview.meeting_url,
        }
        for interview in rows
    ]


@router.post("/{interview_id}/feedback")
async def submit_feedback(
    interview_id: int,
    payload: FeedbackCreate,
    session: AsyncSession = Depends(get_db),
    user=Depends(require_roles("admin", "interviewer", "hr")),
) -> dict:
    interview = (
        await session.execute(
            select(Interview).where(Interview.id == interview_id, Interview.company_id == user.company_id)
        )
    ).scalar_one_or_none()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    feedback = Feedback(
        company_id=user.company_id,
        candidate_id=interview.candidate_id,
        interview_id=interview.id,
        reviewer_id=user.id,
        round_number=interview.round_number,
        rating=payload.rating,
        strengths=payload.strengths,
        concerns=payload.concerns,
        recommendation=payload.recommendation,
        should_advance=payload.should_advance,
    )
    session.add(feedback)
    await session.commit()
    await workflow_service.run_feedback_cycle(
        session,
        interview.candidate_id,
        user.company_id,
        {
            "rating": payload.rating,
            "strengths": payload.strengths,
            "concerns": payload.concerns,
            "recommendation": payload.recommendation,
            "should_advance": payload.should_advance,
        },
    )
    await session.commit()
    return {"status": "saved", "feedback_id": feedback.id}
