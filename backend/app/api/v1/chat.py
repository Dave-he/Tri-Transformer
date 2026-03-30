import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.chat import (
    CreateSessionRequest,
    CreateSessionResponse,
    SendMessageRequest,
    MessageResponse,
    HistoryMessage,
)
from app.services.chat.chat_service import ChatService

router = APIRouter()


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    payload: CreateSessionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = ChatService(db)
    session = await svc.create_session(user_id=current_user.id, title=payload.title)
    return CreateSessionResponse(
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: str,
    payload: SendMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = ChatService(db)
    session = await svc.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    msg = await svc.send_message(
        session_id=session_id,
        user_id=current_user.id,
        kb_id=current_user.kb_id,
        content=payload.content,
    )
    sources = json.loads(msg.sources) if msg.sources else []
    return MessageResponse(
        message_id=msg.id,
        role=msg.role,
        content=msg.content,
        sources=sources,
        hallucination_detected=msg.hallucination_detected,
        created_at=msg.created_at,
    )


@router.get("/sessions/{session_id}/history", response_model=list[HistoryMessage])
async def get_history(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = ChatService(db)
    session = await svc.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    messages = await svc.get_history(session_id, current_user.id)
    return [
        HistoryMessage(
            message_id=m.id,
            role=m.role,
            content=m.content,
            sources=json.loads(m.sources) if m.sources else [],
            created_at=m.created_at,
        )
        for m in messages
    ]
