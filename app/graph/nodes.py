from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.email_agent import EmailAgent
from app.agents.feedback_agent import FeedbackAgent
from app.agents.voice_agent import VoiceSchedulingAgent
from app.graph.state import WorkflowGraphState
from app.models.candidate import Candidate
from app.models.common import CandidateStatusEnum, WorkflowStatusEnum
from app.models.resume import Resume
from app.models.workflow_state import WorkflowState
from app.services.calendar_service import CalendarService
from app.services.email_service import EmailService
from app.services.scoring_service import ScoringService
from app.services.voice_service import VoiceService


class WorkflowNodes:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.scoring_service = ScoringService()
        self.email_agent = EmailAgent()
        self.email_service = EmailService()
        self.voice_agent = VoiceSchedulingAgent()
        self.voice_service = VoiceService()
        self.calendar_service = CalendarService()
        self.feedback_agent = FeedbackAgent()

    async def resume_analysis(self, state: WorkflowGraphState) -> WorkflowGraphState:
        candidate = await self._candidate(state)
        resume = await self._resume(candidate.id)
        await self.scoring_service.score_candidate(self.session, candidate, resume.raw_text)
        candidate.status = CandidateStatusEnum.under_review
        state["stage"] = "resume_scored"
        state["last_action"] = "resume_analysis_completed"
        state["next_action"] = "semantic_match"
        await self._persist(candidate, state)
        return state

    async def semantic_match(self, state: WorkflowGraphState) -> WorkflowGraphState:
        state["stage"] = "ranked"
        state["last_action"] = "semantic_match_completed"
        state["next_action"] = "hitl_gate"
        await self._persist(await self._candidate(state), state)
        return state

    async def hitl_gate(self, state: WorkflowGraphState) -> WorkflowGraphState:
        candidate = await self._candidate(state)
        if candidate.status == CandidateStatusEnum.approved:
            state["status"] = "approved"
            state["stage"] = "approved"
            state["next_action"] = "email_outreach"
        elif candidate.status == CandidateStatusEnum.rejected:
            state["status"] = "completed"
            state["stage"] = "rejected"
            state["next_action"] = "end"
        else:
            state["status"] = "pending"
            state["stage"] = "awaiting_hr"
            state["next_action"] = "await_human"
        state["last_action"] = "hitl_checked"
        await self._persist(candidate, state)
        return state

    async def email_outreach(self, state: WorkflowGraphState) -> WorkflowGraphState:
        candidate = await self._candidate(state)
        draft = await self.email_agent.generate(candidate.full_name, candidate.job.title, candidate.notes or "")
        sent = await self.email_service.send_email(candidate.email, draft.subject, draft.body)
        if not sent:
            state["status"] = "retrying"
            state["stage"] = "email_failed"
            state["next_action"] = "failure_handler"
        else:
            candidate.status = CandidateStatusEnum.email_sent
            state["status"] = "pending"
            state["stage"] = "email_sent"
            state["next_action"] = "voice_schedule"
        state["last_action"] = "email_attempted"
        await self._persist(candidate, state)
        return state

    async def voice_schedule(self, state: WorkflowGraphState) -> WorkflowGraphState:
        candidate = await self._candidate(state)
        planned = await self.voice_agent.plan(candidate.company.timezone, ["10:00", "14:00"])
        success = await self.voice_service.initiate_schedule_call(candidate.phone or "", candidate.id)
        state["schedule"] = {"plan": planned.model_dump()}
        if success:
            candidate.status = CandidateStatusEnum.call_pending
            state["stage"] = "call_pending"
            state["next_action"] = "calendar_booking"
        else:
            state["status"] = "retrying"
            state["stage"] = "call_failed"
            state["next_action"] = "failure_handler"
        state["last_action"] = "voice_scheduling_attempted"
        await self._persist(candidate, state)
        return state

    async def calendar_booking(self, state: WorkflowGraphState) -> WorkflowGraphState:
        candidate = await self._candidate(state)
        interviewer = await self.calendar_service.allocate_interviewer(self.session, candidate.company_id)
        interview = await self.calendar_service.book_slot(
            candidate_id=candidate.id,
            company_id=candidate.company_id,
            interviewer_id=interviewer.id if interviewer else None,
            round_number=state.get("interview_round", 1),
            timezone_name=candidate.company.timezone,
            session=self.session,
        )
        candidate.status = CandidateStatusEnum.scheduled
        state["status"] = "pending"
        state["stage"] = "scheduled"
        state["schedule"] = {"interview_id": interview.id, "meeting_url": interview.meeting_url}
        state["next_action"] = "await_feedback"
        state["last_action"] = "calendar_booked"
        await self._persist(candidate, state)
        return state

    async def feedback_reasoning(self, state: WorkflowGraphState) -> WorkflowGraphState:
        feedback_blob = str(state.get("feedback", {}))
        result = await self.feedback_agent.reason("candidate", state.get("interview_round", 1), feedback_blob)
        state["feedback"] = {
            "summary": result.summary,
            "rationale": result.rationale,
            "should_advance": result.should_advance,
        }
        state["next_action"] = "evaluate_round"
        await self._persist(await self._candidate(state), state)
        return state

    async def evaluate_round(self, state: WorkflowGraphState) -> WorkflowGraphState:
        should_advance = bool(state.get("feedback", {}).get("should_advance"))
        if should_advance:
            state["interview_round"] = state.get("interview_round", 1) + 1
            state["stage"] = "next_round"
            state["next_action"] = "email_outreach"
            state["status"] = "pending"
        else:
            state["stage"] = "workflow_completed"
            state["next_action"] = "end"
            state["status"] = "completed"
        state["last_action"] = "round_evaluated"
        await self._persist(await self._candidate(state), state)
        return state

    async def failure_handler(self, state: WorkflowGraphState) -> WorkflowGraphState:
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        if retry_count >= 3:
            state["status"] = "failed"
            state["stage"] = "failed"
            state["next_action"] = "end"
        else:
            state["status"] = "retrying"
            state["next_action"] = "email_outreach" if state.get("stage") == "call_failed" else "voice_schedule"
        state["last_action"] = "failure_handler_processed"
        await self._persist(await self._candidate(state), state)
        return state

    async def _candidate(self, state: WorkflowGraphState) -> Candidate:
        return (
            await self.session.execute(
                select(Candidate)
                .where(Candidate.id == state["candidate_id"], Candidate.company_id == state["company_id"])
                .options(selectinload(Candidate.job), selectinload(Candidate.company))
            )
        ).scalar_one()

    async def _resume(self, candidate_id: int) -> Resume:
        return (await self.session.execute(select(Resume).where(Resume.candidate_id == candidate_id))).scalar_one()

    async def _persist(self, candidate: Candidate, state: WorkflowGraphState) -> None:
        workflow = (
            await self.session.execute(
                select(WorkflowState).where(
                    WorkflowState.candidate_id == candidate.id,
                    WorkflowState.company_id == candidate.company_id,
                    WorkflowState.interview_round == state.get("interview_round", 1),
                )
            )
        ).scalar_one_or_none()
        if workflow:
            workflow.current_node = state.get("next_action", workflow.current_node)
            workflow.stage = state.get("stage", workflow.stage)
            workflow.status = WorkflowStatusEnum(state.get("status", WorkflowStatusEnum.pending.value))
            workflow.retry_count = state.get("retry_count", workflow.retry_count)
            workflow.last_action = state.get("last_action", workflow.last_action)
            workflow.state_payload = dict(state)
        await self.session.flush()
