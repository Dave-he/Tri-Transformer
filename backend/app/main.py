from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, knowledge, chat, model, train
from app.core.database import create_tables


app = FastAPI(
    title="Tri-Transformer RAG Backend",
    description="Tri-Transformer 可控对话与 RAG 知识库增强系统后端服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(model.router, prefix="/api/v1/model", tags=["model"])
app.include_router(train.router, prefix="/api/v1/train", tags=["train"])


@app.on_event("startup")
async def startup():
    await create_tables()


@app.get("/health")
async def health():
    return {"status": "ok"}
