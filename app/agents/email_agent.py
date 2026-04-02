from pydantic import BaseModel

from app.agents.client import AIClient
from app.agents.prompts import EMAIL_GENERATION_PROMPT


class EmailDraftOutput(BaseModel):
    subject: str
    body: str


class EmailAgent:
    def __init__(self) -> None:
        self.client = AIClient()

    async def generate(self, candidate_name: str, job_title: str, notes: str) -> EmailDraftOutput:
        if self.client.enabled:
            prompt = EMAIL_GENERATION_PROMPT.format(candidate_name=candidate_name, job_title=job_title, notes=notes)
            return await self.client.invoke_structured(prompt, EmailDraftOutput)
        return EmailDraftOutput(
            subject=f"Interview invitation for {job_title}",
            body=(
                f"Hi {candidate_name},\n\n"
                f"Thanks for your interest in the {job_title} role. We'd like to schedule an interview and "
                "will follow up with time options shortly.\n\nBest,\nTalent Team"
            ),
        )

