import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_chunker,
    get_current_user_claims,
    get_db_session,
    get_embedding_model,
    get_extractor,
    get_vector_store,
)
from src.api.schemas.documents import SearchRequest, SearchResponse, SearchResultSchema, UploadResponse
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload(
    file: UploadFile,
    claims: dict = Depends(get_current_user_claims),
    session: AsyncSession = Depends(get_db_session),
) -> UploadResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    content = await file.read()

    storage_dir = Path("storage") / str(tenant_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / file.filename
    storage_path.write_bytes(content)

    use_case = UploadDocument(
        document_repository=PostgresDocumentRepository(session),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
        chunker=get_chunker(),
        extractor=get_extractor(),
    )
    document = await use_case.execute(
        tenant_id=tenant_id, filename=file.filename, content=content, storage_path=str(storage_path)
    )
    await session.commit()

    return UploadResponse(
        id=document.id, filename=document.filename, status=document.status, chunk_count=document.chunk_count
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    claims: dict = Depends(get_current_user_claims),
) -> SearchResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    use_case = SearchDocuments(embedding_model=get_embedding_model(), vector_store=get_vector_store())
    results = await use_case.execute(tenant_id=tenant_id, query=payload.query, top_k=payload.top_k)

    return SearchResponse(
        results=[
            SearchResultSchema(document_id=r.document_id, chunk_id=r.chunk_id, content=r.content, score=r.score)
            for r in results
        ]
    )
