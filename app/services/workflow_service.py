from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.nodes import WorkflowNodes
from app.graph.workflow import build_workflow
from app.models.common import WorkflowStatusEnum
from app.models.workflow_state import WorkflowState


class WorkflowService:
    async def run_candidate_workflow(self, session: AsyncSession, candidate_id: int, company_id: int) -> dict:
        workflow_record = (
            await session.execute(
                select(WorkflowState).where(
                    WorkflowState.candidate_id == candidate_id,
                    WorkflowState.company_id == company_id,
                ).order_by(WorkflowState.interview_round.desc())
            )
        ).scalars().first()
        if workflow_record is None:
            raise ValueError("Workflow state not found for candidate")
        graph = build_workflow(session)
        result = await graph.ainvoke(dict(workflow_record.state_payload))
        workflow_record.state_payload = dict(result)
        workflow_record.current_node = result.get("next_action", workflow_record.current_node)
        workflow_record.stage = result.get("stage", workflow_record.stage)
        workflow_record.status = WorkflowStatusEnum(result.get("status", workflow_record.status.value))
        await session.flush()
        return result

    async def run_feedback_cycle(self, session: AsyncSession, candidate_id: int, company_id: int, feedback: dict) -> dict:
        workflow_record = (
            await session.execute(
                select(WorkflowState).where(
                    WorkflowState.candidate_id == candidate_id,
                    WorkflowState.company_id == company_id,
                ).order_by(WorkflowState.interview_round.desc())
            )
        ).scalars().first()
        if workflow_record is None:
            raise ValueError("Workflow state not found for candidate")

        state = dict(workflow_record.state_payload)
        state["feedback"] = feedback
        nodes = WorkflowNodes(session)
        state = await nodes.feedback_reasoning(state)
        state = await nodes.evaluate_round(state)

        if state.get("next_action") == "email_outreach":
            next_round = state.get("interview_round", workflow_record.interview_round)
            workflow_record.status = WorkflowStatusEnum.completed
            workflow_record.current_node = "end"
            workflow_record.stage = "round_completed"
            next_workflow = WorkflowState(
                company_id=company_id,
                candidate_id=candidate_id,
                interview_round=next_round,
                current_node="email_outreach",
                stage=state.get("stage", "next_round"),
                status=WorkflowStatusEnum.pending,
                retry_count=0,
                last_action="new_round_initialized",
                state_payload=dict(state),
            )
            session.add(next_workflow)
            state = await nodes.email_outreach(state)
            state = await nodes.voice_schedule(state)
            state = await nodes.calendar_booking(state)
            workflow_record = next_workflow

        workflow_record.state_payload = dict(state)
        workflow_record.current_node = state.get("next_action", workflow_record.current_node)
        workflow_record.stage = state.get("stage", workflow_record.stage)
        workflow_record.status = WorkflowStatusEnum(state.get("status", workflow_record.status.value))
        await session.flush()
        return state
