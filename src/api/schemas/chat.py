import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Bounded so a multi-megabyte "question" can't be embedded and forwarded
    # into a Claude request at the caller's discretion.
    question: str = Field(..., max_length=4000)


class ChatSourceSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSourceSchema]
