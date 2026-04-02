import hashlib
import math

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.client import AIClient
from app.models.embedding import Embedding


class EmbeddingService:
    def __init__(self) -> None:
        self.client = AIClient()
        self.dimension = 1536

    async def embed_text(self, text: str) -> list[float]:
        if self.client.enabled:
            return await self.client.embed_query(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(self.dimension)]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if not norm_a or not norm_b:
            return 0.0
        return numerator / (norm_a * norm_b)

    async def store_embedding(
        self,
        session: AsyncSession,
        *,
        company_id: int,
        source_type: str,
        source_id: str,
        vector: list[float],
        job_id: int | None = None,
        candidate_id: int | None = None,
    ) -> Embedding:
        embedding = Embedding(
            company_id=company_id,
            source_type=source_type,
            source_id=source_id,
            embedding=vector,
            job_id=job_id,
            candidate_id=candidate_id,
        )
        session.add(embedding)
        await session.flush()
        return embedding

