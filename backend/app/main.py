from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1 import auth, knowledge, chat, model, train, stream, webrtc, metrics
from app.core.database import create_tables
from app.core.config import settings
from app.core.logging import request_logging_middleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(
    title="Tri-Transformer RAG Backend",
    description="Tri-Transformer 可控对话与 RAG 知识库增强系统后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

app.add_middleware(SecurityHeadersMiddleware)
app.middleware("http")(request_logging_middleware)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(model.router, prefix="/api/v1/model", tags=["model"])
app.include_router(train.router, prefix="/api/v1/train", tags=["train"])
app.include_router(stream.router, prefix="/api/v1/model", tags=["stream"])
app.include_router(webrtc.router, prefix="/api/v1/webrtc", tags=["webrtc"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


@app.get("/health")
async def health():
    return {"status": "ok"}
