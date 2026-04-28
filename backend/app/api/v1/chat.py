import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.chat_session import ChatMessage
from app.schemas.chat import (
    CreateSessionRequest,
    CreateSessionResponse,
    SendMessageRequest,
    MessageResponse,
    SendMessageResponse,
    HistoryMessage,
    ConversationItem,
    ConversationListResponse,
    PaginationInfo,
    SessionDeleteResponse,
)
from app.services.chat.chat_service import ChatService

router = APIRouter()


@router.get("/sessions", response_model=ConversationListResponse)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query("all"),
):
    svc = ChatService(db)
    sessions, total, total_pages = await svc.list_sessions(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
    )
    conversations = []
    for s in sessions:
        count_result = await db.execute(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s.id)
        )
        msg_count = count_result.scalar() or 0
        conversations.append(
            ConversationItem(
                id=s.id,
                title=s.title,
                status=s.status,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=msg_count,
            )
        )
    return ConversationListResponse(
        conversations=conversations,
        pagination=PaginationInfo(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        ),
    )


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    payload: CreateSessionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = ChatService(db)
    session = await svc.create_session(user_id=current_user.id, title=payload.title)
    return CreateSessionResponse(
        id=session.id,
        session_id=session.id,
        title=session.title,
        created_at=session.created_at,
    )


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
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
    return SendMessageResponse(
        message=MessageResponse(
            message_id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=sources,
            hallucination_detected=msg.hallucination_detected,
            created_at=msg.created_at,
        )
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


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    svc = ChatService(db)
    session = await svc.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    await svc.delete_session(session_id, current_user.id)
    return SessionDeleteResponse(message="Session deleted")
