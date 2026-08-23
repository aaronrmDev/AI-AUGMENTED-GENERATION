import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_chunker,
    get_current_user_claims,
    get_db_session,
    get_embedding_model,
    get_extractor,
    get_file_storage,
    get_vector_store,
)
from src.api.schemas.documents import (
    SearchRequest,
    SearchResponse,
    SearchResultSchema,
    UploadResponse,
)
from src.rag.application.search_documents import SearchDocuments
from src.rag.application.upload_document import UploadDocument
from src.rag.infrastructure.postgres_document_repository import PostgresDocumentRepository

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("", response_model=UploadResponse, status_code=201)
async def upload(
    file: UploadFile,
    claims: dict[str, Any] = Depends(get_current_user_claims),
    session: AsyncSession = Depends(get_db_session),
) -> UploadResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        # A cheap ceiling, not a full fix: the whole body is already in memory
        # by the time this runs, because the handler still reads it eagerly.
        # It bounds what a single request can cost downstream (chunking and
        # embedding are both linear in the byte count and both run inline on
        # the event loop) until that work is moved off the request path.
        raise HTTPException(status_code=413, detail="File too large")

    # A multipart part is not required to carry a filename. Falling back to a
    # placeholder keeps the extension-less case on the domain's own path --
    # TextExtractor raises UnsupportedFileType for it, which the handler in
    # exception_handlers.py turns into a 422 -- rather than a 500 on a None
    # deref inside the extractor.
    filename = file.filename or "upload"

    use_case = UploadDocument(
        document_repository=PostgresDocumentRepository(session),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
        chunker=get_chunker(),
        extractor=get_extractor(),
        file_storage=get_file_storage(),
    )
    document = await use_case.execute(tenant_id=tenant_id, filename=filename, content=content)
    await session.commit()

    return UploadResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        chunk_count=document.chunk_count,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    claims: dict[str, Any] = Depends(get_current_user_claims),
) -> SearchResponse:
    tenant_id = uuid.UUID(claims["tenant_id"])
    use_case = SearchDocuments(
        embedding_model=get_embedding_model(), vector_store=get_vector_store()
    )
    results = await use_case.execute(tenant_id=tenant_id, query=payload.query, top_k=payload.top_k)

    return SearchResponse(
        results=[
            SearchResultSchema(
                document_id=r.document_id, chunk_id=r.chunk_id, content=r.content, score=r.score
            )
            for r in results
        ]
    )
