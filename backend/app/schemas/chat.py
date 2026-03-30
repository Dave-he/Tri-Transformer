from datetime import datetime
from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class CreateSessionResponse(BaseModel):
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
