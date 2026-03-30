import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.core.security import create_access_token, hash_password
from app.models.user import User


DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def fc_engine():
    engine = create_async_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def fc_client(fc_engine):
    session_factory = async_sessionmaker(
        fc_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])
    with patch(
        "app.services.chat.chat_service.ChromaVectorStore",
        return_value=AsyncMock(),
    ), patch(
        "app.services.chat.chat_service.HybridRetriever",
        return_value=mock_retriever,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def fc_user(fc_engine):
    session_factory = async_sessionmaker(
        fc_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        user = User(
            username="fc_user",
            email="fc@example.com",
            hashed_password=hash_password("password"),
            kb_id="kb-fc-001",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest.fixture
def fc_token(fc_user):
    return create_access_token({"sub": str(fc_user.id), "kb_id": fc_user.kb_id})


class TestFactChecker:
    def test_check_same_text_high_score(self):
        from app.services.model.fact_checker import FactChecker

        fc = FactChecker()
        text = "The capital of France is Paris."
        result = fc.check(generated=text, contexts=[text])
        assert result.score >= 0.9
        assert result.hallucination_detected is False

    def test_check_different_text_low_score(self):
        from app.services.model.fact_checker import FactChecker

        fc = FactChecker(threshold=0.9)
        result = fc.check(
            generated="The sky is green and grass is blue.",
            contexts=["Water is H2O. Dogs are mammals."],
        )
        assert result.hallucination_detected is True

    def test_check_returns_fact_check_result(self):
        from app.services.model.fact_checker import FactChecker, FactCheckResult

        fc = FactChecker()
        result = fc.check(generated="test", contexts=["test"])
        assert isinstance(result, FactCheckResult)
        assert hasattr(result, "score")
        assert hasattr(result, "hallucination_detected")
        assert 0.0 <= result.score <= 1.0

    def test_check_empty_contexts(self):
        from app.services.model.fact_checker import FactChecker

        fc = FactChecker()
        result = fc.check(generated="some text", contexts=[])
        assert isinstance(result.score, float)
        assert isinstance(result.hallucination_detected, bool)

    def test_threshold_boundary(self):
        from app.services.model.fact_checker import FactChecker

        fc_low = FactChecker(threshold=0.0)
        result = fc_low.check(generated="abc", contexts=["def"])
        assert result.hallucination_detected is False

        fc_high = FactChecker(threshold=1.0)
        result2 = fc_high.check(generated="abc", contexts=["def"])
        assert result2.hallucination_detected is True


class TestChatAPIHallucinationField:
    async def test_send_message_includes_hallucination_detected(self, fc_client, fc_token):
        resp = await fc_client.post(
            "/api/v1/chat/sessions",
            json={"title": "FC Test"},
            headers={"Authorization": f"Bearer {fc_token}"},
        )
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]

        resp2 = await fc_client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json={"content": "Hello"},
            headers={"Authorization": f"Bearer {fc_token}"},
        )
        assert resp2.status_code == 200
        body = resp2.json()
        assert "hallucination_detected" in body
        assert isinstance(body["hallucination_detected"], bool)
