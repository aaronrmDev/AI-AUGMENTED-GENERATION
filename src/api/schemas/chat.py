import uuid

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatSourceSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSourceSchema]
