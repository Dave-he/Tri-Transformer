from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    progress: int
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    type: str = ""
    size: int = 0
    status: str
    chunk_count: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str


class SearchResultItem(BaseModel):
    text: str
    score: float
    rerank_score: Optional[float] = None
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentDeleteResponse(BaseModel):
    message: str
