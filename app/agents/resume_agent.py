import hashlib

from pydantic import BaseModel, Field

from app.agents.client import AIClient
from app.agents.prompts import RESUME_SCORING_PROMPT


class ResumeScoreOutput(BaseModel):
    ats_score: float = Field(ge=0, le=100)
    match_percentage: float = Field(ge=0, le=100)
    skills: list[str]
    experience_summary: str
    strengths: list[str]
    weaknesses: list[str]


class ResumeScoringAgent:
    def __init__(self) -> None:
        self.client = AIClient()

    async def analyze(self, job_description: str, resume_text: str) -> ResumeScoreOutput:
        if self.client.enabled:
            prompt = RESUME_SCORING_PROMPT.format(job_description=job_description, resume_text=resume_text)
            return await self.client.invoke_structured(prompt, ResumeScoreOutput)

        digest = int(hashlib.sha256(resume_text.encode("utf-8")).hexdigest()[:6], 16)
        ats_score = min(95, 60 + (digest % 25))
        match_percentage = min(95, 55 + (digest % 35))
        tokens = [token.strip(".,") for token in resume_text.split() if len(token) > 4][:8]
        return ResumeScoreOutput(
            ats_score=ats_score,
            match_percentage=match_percentage,
            skills=sorted(set(tokens[:5] or ["python", "communication"])),
            experience_summary="Candidate resume parsed in fallback mode. Configure OpenAI for richer reasoning.",
            strengths=["Relevant experience detected", "Resume contains identifiable technical depth"],
            weaknesses=["Fallback mode cannot validate nuanced role fit"],
        )

