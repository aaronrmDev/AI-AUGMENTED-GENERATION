import uuid

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    # Bounded so a caller can't ask Qdrant for a million neighbours in one
    # request; rejected at the schema as a 422 rather than absorbed downstream.
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResultSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultSchema]
