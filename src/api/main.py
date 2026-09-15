from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.rate_limit import RateLimitHeadersMiddleware
from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.documents import router as documents_router
from src.api.routers.sessions import router as sessions_router

app = FastAPI(title="Unified RAG x CAG x MAG AI System")
register_exception_handlers(app)
app.add_middleware(RateLimitHeadersMiddleware)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(sessions_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def ensure_qdrant_collection() -> None:
    from src.api.dependencies import get_vector_store

    await get_vector_store().ensure_collection()


@app.on_event("shutdown")
async def release_process_resources() -> None:
    from src.api.dependencies import close_redis_clients, get_unified_pipeline

    # Only a pipeline some request built has anything to drain; don't build one now.
    if get_unified_pipeline.cache_info().currsize:
        await get_unified_pipeline().cascade.drain()
    await close_redis_clients()
