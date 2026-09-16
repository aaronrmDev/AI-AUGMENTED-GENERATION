from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware.max_body_size import MaxBodySizeMiddleware
from src.api.middleware.security_headers import SecurityHeadersMiddleware
from src.api.rate_limit import RateLimitHeadersMiddleware
from src.api.routers.auth import router as auth_router
from src.api.routers.chat import router as chat_router
from src.api.routers.data_sources import router as data_sources_router
from src.api.routers.documents import router as documents_router
from src.api.routers.sessions import router as sessions_router
from src.api.security_logging import configure_security_logging

app = FastAPI(title="Unified RAG x CAG x MAG AI System")
register_exception_handlers(app)
app.add_middleware(RateLimitHeadersMiddleware)
app.add_middleware(
    MaxBodySizeMiddleware,
    default_max_bytes=16 * 1024,
    path_overrides={"/documents": 11 * 1024 * 1024, "/data-sources": 11 * 1024 * 1024},
)
app.add_middleware(SecurityHeadersMiddleware)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(data_sources_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def configure_security_logging_on_startup() -> None:
    configure_security_logging()


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
