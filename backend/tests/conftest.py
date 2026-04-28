import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.core.security import create_access_token, hash_password
from app.models.user import User
from app.models.document import Document
from app.models.chat_session import ChatSession, ChatMessage
from app.models.train_job import TrainJob


DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_engine():
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


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine):
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(test_engine):
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    from app.api.v1 import auth as auth_module
    original_enabled = auth_module.limiter.enabled
    auth_module.limiter.enabled = False

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    with patch(
        "app.services.chat.chat_service.ChromaVectorStore",
        return_value=AsyncMock(),
    ), patch(
        "app.services.chat.chat_service.HybridRetriever",
        return_value=mock_retriever,
    ), patch(
        "app.api.v1.train._run_training",
    ), patch(
        "app.api.v1.knowledge.get_vector_store",
        return_value=AsyncMock(
            delete_document=AsyncMock(),
            query=AsyncMock(return_value=[]),
        ),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    auth_module.limiter.enabled = original_enabled
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session):
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hash_password("testpassword"),
        kb_id="kb-test-001",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_token(test_user):
    return create_access_token({"sub": str(test_user.id), "kb_id": test_user.kb_id})


@pytest.fixture(scope="function")
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
