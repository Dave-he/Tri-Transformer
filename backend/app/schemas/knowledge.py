from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: Optional[dict] = None


class SearchResultItem(BaseModel):
    text: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
