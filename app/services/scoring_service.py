from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.resume_agent import ResumeScoringAgent
from app.models.candidate import Candidate
from app.models.job import Job
from app.models.score import Score
from app.services.embedding_service import EmbeddingService


class ScoringService:
    def __init__(self) -> None:
        self.resume_agent = ResumeScoringAgent()
        self.embedding_service = EmbeddingService()

    async def score_candidate(self, session: AsyncSession, candidate: Candidate, resume_text: str) -> Score:
        job = (
            await session.execute(select(Job).where(Job.id == candidate.job_id, Job.company_id == candidate.company_id))
        ).scalar_one()

        analysis = await self.resume_agent.analyze(job.description, resume_text)
        job_embedding = await self.embedding_service.embed_text(job.description)
        resume_embedding = await self.embedding_service.embed_text(resume_text)
        embedding_score = max(0.0, self.embedding_service.cosine_similarity(job_embedding, resume_embedding) * 100)
        llm_score = float(analysis.match_percentage)
        final_score = round((llm_score * 0.6) + (embedding_score * 0.4), 2)

        await self.embedding_service.store_embedding(
            session,
            company_id=candidate.company_id,
            source_type="job",
            source_id=str(job.id),
            vector=job_embedding,
            job_id=job.id,
        )
        await self.embedding_service.store_embedding(
            session,
            company_id=candidate.company_id,
            source_type="candidate",
            source_id=str(candidate.id),
            vector=resume_embedding,
            candidate_id=candidate.id,
        )

        score = Score(
            company_id=candidate.company_id,
            candidate_id=candidate.id,
            ats_score=analysis.ats_score,
            llm_score=llm_score,
            embedding_score=round(embedding_score, 2),
            match_percentage=analysis.match_percentage,
            final_score=final_score,
            reasoning={"skills": analysis.skills, "strengths": analysis.strengths, "weaknesses": analysis.weaknesses},
            experience_summary=analysis.experience_summary,
        )
        candidate.final_score = final_score
        if analysis.skills and not candidate.current_title:
            candidate.current_title = analysis.skills[0]
        session.add(score)
        await session.flush()
        return score

