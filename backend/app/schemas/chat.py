from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class CreateSessionResponse(BaseModel):
    id: str
    session_id: str
    title: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    sources: list[dict]
    hallucination_detected: bool = False
    created_at: datetime


class HistoryMessage(BaseModel):
    message_id: str
    role: str
    content: str
    sources: list[dict]
    created_at: datetime


class ConversationItem(BaseModel):
    id: str
    title: str
    status: str = "active"
    created_at: datetime
    updated_at: Optional[datetime] = None
    message_count: int = 0


class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class SendMessageResponse(BaseModel):
    message: MessageResponse


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]
    pagination: PaginationInfo


class SessionDeleteResponse(BaseModel):
    message: str
