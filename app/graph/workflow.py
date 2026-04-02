from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.nodes import WorkflowNodes
from app.graph.state import WorkflowGraphState


def _next_after_hitl(state: WorkflowGraphState) -> str:
    return "email_outreach" if state.get("next_action") == "email_outreach" else END


def _next_after_email(state: WorkflowGraphState) -> str:
    return "failure_handler" if state.get("next_action") == "failure_handler" else "voice_schedule"


def _next_after_voice(state: WorkflowGraphState) -> str:
    return "failure_handler" if state.get("next_action") == "failure_handler" else "calendar_booking"


def _next_after_failure(state: WorkflowGraphState) -> str:
    action = state.get("next_action")
    return action if action in {"voice_schedule", "email_outreach"} else END


def _next_after_evaluation(state: WorkflowGraphState) -> str:
    return "email_outreach" if state.get("next_action") == "email_outreach" else END


def build_workflow(session: AsyncSession):
    nodes = WorkflowNodes(session)
    graph = StateGraph(WorkflowGraphState)
    graph.add_node("resume_analysis", nodes.resume_analysis)
    graph.add_node("semantic_match", nodes.semantic_match)
    graph.add_node("hitl_gate", nodes.hitl_gate)
    graph.add_node("email_outreach", nodes.email_outreach)
    graph.add_node("voice_schedule", nodes.voice_schedule)
    graph.add_node("calendar_booking", nodes.calendar_booking)
    graph.add_node("feedback_reasoning", nodes.feedback_reasoning)
    graph.add_node("evaluate_round", nodes.evaluate_round)
    graph.add_node("failure_handler", nodes.failure_handler)

    graph.add_edge(START, "resume_analysis")
    graph.add_edge("resume_analysis", "semantic_match")
    graph.add_edge("semantic_match", "hitl_gate")
    graph.add_conditional_edges("hitl_gate", _next_after_hitl, {"email_outreach": "email_outreach", END: END})
    graph.add_conditional_edges("email_outreach", _next_after_email, {"voice_schedule": "voice_schedule", "failure_handler": "failure_handler"})
    graph.add_conditional_edges("voice_schedule", _next_after_voice, {"calendar_booking": "calendar_booking", "failure_handler": "failure_handler"})
    graph.add_edge("calendar_booking", END)
    graph.add_conditional_edges(
        "failure_handler",
        _next_after_failure,
        {"voice_schedule": "voice_schedule", "email_outreach": "email_outreach", END: END},
    )
    graph.add_edge("feedback_reasoning", "evaluate_round")
    graph.add_conditional_edges("evaluate_round", _next_after_evaluation, {"email_outreach": "email_outreach", END: END})
    return graph.compile()

