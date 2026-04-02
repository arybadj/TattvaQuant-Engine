from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.common import DecisionEnum
from app.schemas.candidate import CandidateDetailResponse, CandidateUploadResponse, DecisionCreate, DecisionResponse
from app.services.auth import get_current_user, require_roles
from app.services.candidate_service import CandidateService
from app.services.workflow_service import WorkflowService
from app.workers.tasks import process_candidate_workflow_task

router = APIRouter(prefix="/candidates", tags=["candidates"])
candidate_service = CandidateService()
workflow_service = WorkflowService()


@router.post("/upload", response_model=CandidateUploadResponse)
async def upload_candidate(
    job_id: int = Form(...),
    full_name: str = Form(...),
    email: str = Form(...),
    phone: str | None = Form(default=None),
    resume: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    user=Depends(require_roles("admin", "hr")),
) -> CandidateUploadResponse:
    candidate, workflow_state = await candidate_service.create_candidate(
        session=session,
        company_id=user.company_id,
        job_id=job_id,
        full_name=full_name,
        email=email,
        phone=phone,
        resume_file=resume,
    )
    await session.commit()
    process_candidate_workflow_task.delay(candidate.id, user.company_id)
    return CandidateUploadResponse(candidate_id=candidate.id, workflow_state_id=workflow_state.id, status="queued")


@router.get("/{candidate_id}", response_model=CandidateDetailResponse)
async def get_candidate(
    candidate_id: int,
    session: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
) -> CandidateDetailResponse:
    return await candidate_service.get_candidate_detail(session, user.company_id, candidate_id)


@router.post("/{candidate_id}/decision", response_model=DecisionResponse)
async def apply_decision(
    candidate_id: int,
    payload: DecisionCreate,
    session: AsyncSession = Depends(get_db),
    user=Depends(require_roles("admin", "hr")),
) -> DecisionResponse:
    decision = await candidate_service.apply_decision(
        session=session,
        company_id=user.company_id,
        candidate_id=candidate_id,
        reviewer_id=user.id,
        decision=payload.decision.value,
        notes=payload.notes,
    )
    await session.commit()
    if payload.decision == DecisionEnum.approved:
        await workflow_service.run_candidate_workflow(session, candidate_id, user.company_id)
        await session.commit()
    return DecisionResponse.model_validate(decision)
