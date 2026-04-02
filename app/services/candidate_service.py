from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.candidate import Candidate
from app.models.common import CandidateStatusEnum, DecisionEnum, WorkflowStatusEnum
from app.models.hr_decision import HRDecision
from app.models.job import Job
from app.models.resume import Resume
from app.models.workflow_state import WorkflowState
from app.schemas.candidate import CandidateDetailResponse

settings = get_settings()


class CandidateService:
    async def create_candidate(
        self,
        *,
        session: AsyncSession,
        company_id: int,
        job_id: int,
        full_name: str,
        email: str,
        phone: str | None,
        resume_file: UploadFile,
    ) -> tuple[Candidate, WorkflowState]:
        await self._validate_upload(resume_file)
        job = (await session.execute(select(Job).where(Job.id == job_id, Job.company_id == company_id))).scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        resume_text = await self._extract_text(resume_file)
        candidate = Candidate(
            company_id=company_id,
            job_id=job_id,
            full_name=full_name,
            email=email,
            phone=phone,
            status=CandidateStatusEnum.uploaded,
        )
        session.add(candidate)
        await session.flush()

        session.add(
            Resume(
                company_id=company_id,
                candidate_id=candidate.id,
                filename=resume_file.filename or "resume.pdf",
                content_type=resume_file.content_type or "application/pdf",
                raw_text=resume_text,
                structured_data={"word_count": len(resume_text.split())},
            )
        )
        workflow_state = WorkflowState(
            company_id=company_id,
            candidate_id=candidate.id,
            interview_round=1,
            current_node="resume_analysis",
            stage="uploaded",
            status=WorkflowStatusEnum.pending,
            state_payload={
                "candidate_id": candidate.id,
                "company_id": company_id,
                "status": "pending",
                "stage": "uploaded",
                "interview_round": 1,
                "retry_count": 0,
                "last_action": "resume_uploaded",
                "schedule": {},
                "feedback": {},
                "next_action": "resume_analysis",
            },
        )
        session.add(workflow_state)
        await session.flush()
        return candidate, workflow_state

    async def get_candidate_detail(self, session: AsyncSession, company_id: int, candidate_id: int) -> CandidateDetailResponse:
        candidate = (
            await session.execute(
                select(Candidate)
                .where(Candidate.id == candidate_id, Candidate.company_id == company_id)
                .options(
                    selectinload(Candidate.resume),
                    selectinload(Candidate.scores),
                    selectinload(Candidate.decisions),
                    selectinload(Candidate.interviews),
                )
            )
        ).scalar_one_or_none()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        latest_score = candidate.scores[-1] if candidate.scores else None
        latest_decision = candidate.decisions[-1] if candidate.decisions else None
        return CandidateDetailResponse(
            id=candidate.id,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            company_id=candidate.company_id,
            job_id=candidate.job_id,
            full_name=candidate.full_name,
            email=candidate.email,
            phone=candidate.phone,
            current_title=candidate.current_title,
            location=candidate.location,
            status=candidate.status,
            final_score=candidate.final_score,
            notes=candidate.notes,
            resume=candidate.resume,
            latest_score=latest_score,
            latest_decision=latest_decision,
            interviews=list(candidate.interviews),
        )

    async def apply_decision(
        self,
        *,
        session: AsyncSession,
        company_id: int,
        candidate_id: int,
        reviewer_id: int,
        decision: str,
        notes: str | None,
    ) -> HRDecision:
        candidate = (
            await session.execute(select(Candidate).where(Candidate.id == candidate_id, Candidate.company_id == company_id))
        ).scalar_one_or_none()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        if decision == DecisionEnum.approved.value:
            candidate.status = CandidateStatusEnum.approved
        elif decision == DecisionEnum.rejected.value:
            candidate.status = CandidateStatusEnum.rejected
        else:
            candidate.status = CandidateStatusEnum.under_review
        hr_decision = HRDecision(
            company_id=company_id,
            candidate_id=candidate_id,
            reviewer_id=reviewer_id,
            decision=decision,
            notes=notes,
        )
        session.add(hr_decision)
        await session.flush()
        return hr_decision

    async def _validate_upload(self, resume_file: UploadFile) -> None:
        content_type = resume_file.content_type or ""
        if content_type not in {"application/pdf", "text/plain"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

        body = await resume_file.read()
        await resume_file.seek(0)
        if len(body) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File exceeds maximum size")

    async def _extract_text(self, resume_file: UploadFile) -> str:
        body = await resume_file.read()
        await resume_file.seek(0)
        if (resume_file.content_type or "").startswith("text/"):
            return body.decode("utf-8", errors="ignore")

        reader = PdfReader(BytesIO(body))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise HTTPException(status_code=400, detail="Resume text extraction failed")
        return text
