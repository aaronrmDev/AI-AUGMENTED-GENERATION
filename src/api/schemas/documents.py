import uuid

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultSchema]
