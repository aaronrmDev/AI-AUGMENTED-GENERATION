from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routers.auth import RateLimitHeadersMiddleware
from src.api.routers.auth import router as auth_router

app = FastAPI(title="Unified RAG x CAG x MAG AI System")
register_exception_handlers(app)
app.add_middleware(RateLimitHeadersMiddleware)
app.include_router(auth_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
