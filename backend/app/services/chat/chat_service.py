import json
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.chat_session import ChatSession, ChatMessage
from app.services.rag.embedder import get_embedder
from app.services.rag.vector_store import ChromaVectorStore
from app.services.rag.retriever import HybridRetriever
from app.services.model.mock_inference import get_inference_service
from app.services.model.fact_checker import FactChecker
from app.core.config import settings


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, user_id: int, title: str) -> ChatSession:
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session(self, session_id: str, user_id: int) -> Optional[ChatSession]:
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        return session

    async def send_message(
        self,
        session_id: str,
        user_id: int,
        kb_id: str,
        content: str,
    ) -> ChatMessage:
        history_messages = await self.get_history(session_id, user_id)
        history = [
            {"role": m.role, "content": m.content}
            for m in history_messages
        ]

        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=content,
            sources="[]",
        )
        self.db.add(user_msg)
        await self.db.commit()

        try:
            embedder = get_embedder()
            store = ChromaVectorStore(embedder=embedder)
            retriever = HybridRetriever(vector_store=store)
            retrieved = await retriever.retrieve(
                query=content,
                kb_id=kb_id,
                top_k=settings.top_k_rerank,
            )
        except Exception:
            retrieved = []

        context_texts = [r["text"] for r in retrieved]
        sources = [
            {"text": r["text"][:100], "score": r.get("score", 0.0)}
            for r in retrieved
        ]

        inference_svc = get_inference_service()
        result = await inference_svc.infer(
            query=content,
            context=context_texts,
            history=history,
        )

        fact_checker = FactChecker()
        fact_result = fact_checker.check(
            generated=result["text"],
            contexts=context_texts,
        )

        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=result["text"],
            sources=json.dumps(sources, ensure_ascii=False),
            hallucination_detected=fact_result.hallucination_detected,
        )
        self.db.add(assistant_msg)
        await self.db.commit()
        await self.db.refresh(assistant_msg)
        return assistant_msg

    async def get_history(
        self,
        session_id: str,
        user_id: int,
    ) -> list[ChatMessage]:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return result.scalars().all()
