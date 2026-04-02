from typing import TypeVar

from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()
SchemaType = TypeVar("SchemaType", bound=BaseModel)


class AIClient:
    def __init__(self) -> None:
        self.enabled = bool(settings.openai_api_key)
        if self.enabled:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            self.chat = ChatOpenAI(api_key=settings.openai_api_key, model=settings.openai_model, temperature=0)
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        else:
            self.chat = None
            self.embeddings = None

    async def invoke_structured(self, prompt: str, schema: type[SchemaType]) -> SchemaType:
        if not self.enabled or self.chat is None:
            raise RuntimeError("OpenAI integration is not configured.")
        chain = self.chat.with_structured_output(schema)
        return await chain.ainvoke(prompt)

    async def embed_query(self, text: str) -> list[float]:
        if not self.enabled or self.embeddings is None:
            raise RuntimeError("OpenAI embeddings are not configured.")
        return await self.embeddings.aembed_query(text)

