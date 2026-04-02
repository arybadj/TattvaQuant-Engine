from pydantic import BaseModel

from app.agents.client import AIClient
from app.agents.prompts import FEEDBACK_REASONING_PROMPT


class FeedbackReasoningOutput(BaseModel):
    summary: str
    rationale: list[str]
    should_advance: bool


class FeedbackAgent:
    def __init__(self) -> None:
        self.client = AIClient()

    async def reason(self, candidate_name: str, round_number: int, feedback_blob: str) -> FeedbackReasoningOutput:
        if self.client.enabled:
            prompt = FEEDBACK_REASONING_PROMPT.format(
                candidate_name=candidate_name,
                round_number=round_number,
                feedback_blob=feedback_blob,
            )
            return await self.client.invoke_structured(prompt, FeedbackReasoningOutput)
        return FeedbackReasoningOutput(
            summary="Feedback processed in fallback mode.",
            rationale=["Configure OpenAI for richer hiring rationale synthesis."],
            should_advance="strong" in feedback_blob.lower() or "advance" in feedback_blob.lower(),
        )
