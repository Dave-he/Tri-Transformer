import pytest
from app.models.chat_session import ChatSession
from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_create_session(client, auth_headers):
    response = await client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": "Test Session"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["title"] == "Test Session"


@pytest.mark.asyncio
async def test_send_message_returns_reply(client, auth_headers):
    session_resp = await client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": "Chat Session"},
    )
    session_id = session_resp.json()["session_id"]
    response = await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=auth_headers,
        json={"content": "What is Tri-Transformer?"},
    )
    assert response.status_code == 200
    data = response.json()
    msg = data["message"]
    assert "content" in msg
    assert "sources" in msg
    assert isinstance(msg["sources"], list)
    assert len(msg["content"]) > 0


@pytest.mark.asyncio
async def test_send_message_includes_sources(client, auth_headers):
    session_resp = await client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": "Sources Session"},
    )
    session_id = session_resp.json()["session_id"]
    response = await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=auth_headers,
        json={"content": "Tell me about RAG."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data["message"]


@pytest.mark.asyncio
async def test_get_history(client, auth_headers):
    session_resp = await client.post(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        json={"title": "History Session"},
    )
    session_id = session_resp.json()["session_id"]
    await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=auth_headers,
        json={"content": "First message"},
    )
    await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=auth_headers,
        json={"content": "Second message"},
    )
    history_resp = await client.get(
        f"/api/v1/chat/sessions/{session_id}/history",
        headers=auth_headers,
    )
    assert history_resp.status_code == 200
    messages = history_resp.json()
    assert isinstance(messages, list)
    assert len(messages) >= 2


@pytest.mark.asyncio
async def test_cross_user_session_forbidden(client):
    user1_reg = await client.post(
        "/api/v1/auth/register",
        json={"username": "chat_user1", "email": "chatuser1@test.com", "password": "pass"},
    )
    user1_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "chat_user1", "password": "pass"},
    )
    headers1 = {"Authorization": f"Bearer {user1_login.json()['access_token']}"}

    user2_reg = await client.post(
        "/api/v1/auth/register",
        json={"username": "chat_user2", "email": "chatuser2@test.com", "password": "pass"},
    )
    user2_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "chat_user2", "password": "pass"},
    )
    headers2 = {"Authorization": f"Bearer {user2_login.json()['access_token']}"}

    session_resp = await client.post(
        "/api/v1/chat/sessions",
        headers=headers1,
        json={"title": "Private Session"},
    )
    session_id = session_resp.json()["session_id"]

    response = await client.get(
        f"/api/v1/chat/sessions/{session_id}/history",
        headers=headers2,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_session_not_found(client, auth_headers):
    response = await client.get(
        "/api/v1/chat/sessions/nonexistent-session/history",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_unauthorized(client):
    response = await client.post(
        "/api/v1/chat/sessions",
        json={"title": "Unauthorized"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_sessions_empty(client, auth_headers):
    response = await client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["conversations"] == []
    assert data["pagination"]["total"] == 0
    assert data["pagination"]["total_pages"] == 0


@pytest.mark.asyncio
async def test_list_sessions_returns_user_sessions(client, auth_headers):
    for i in range(3):
        await client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": f"Session {i}"},
        )
    response = await client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["conversations"]) == 3
    assert data["pagination"]["total"] == 3
    for conv in data["conversations"]:
        assert "id" in conv
        assert "title" in conv
        assert "status" in conv
        assert "created_at" in conv
        assert "updated_at" in conv
        assert "message_count" in conv


@pytest.mark.asyncio
async def test_list_sessions_user_isolation(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "listuser1", "email": "listuser1@test.com", "password": "pass"},
    )
    user1_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "listuser1", "password": "pass"},
    )
    headers1 = {"Authorization": f"Bearer {user1_login.json()['access_token']}"}
    await client.post(
        "/api/v1/auth/register",
        json={"username": "listuser2", "email": "listuser2@test.com", "password": "pass"},
    )
    user2_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "listuser2", "password": "pass"},
    )
    headers2 = {"Authorization": f"Bearer {user2_login.json()['access_token']}"}

    await client.post(
        "/api/v1/chat/sessions",
        headers=headers1,
        json={"title": "User1 Session"},
    )
    resp1 = await client.get("/api/v1/chat/sessions", headers=headers1)
    resp2 = await client.get("/api/v1/chat/sessions", headers=headers2)
    assert resp1.json()["pagination"]["total"] == 1
    assert resp2.json()["pagination"]["total"] == 0


@pytest.mark.asyncio
async def test_list_sessions_pagination(client, auth_headers):
    for i in range(5):
        await client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers,
            json={"title": f"Page Session {i}"},
        )
    response = await client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        params={"page": 1, "page_size": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["conversations"]) == 2
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 2
    assert data["pagination"]["total"] == 5
    assert data["pagination"]["total_pages"] == 3


@pytest.mark.asyncio
async def test_list_sessions_status_filter(client, auth_headers, db_session, test_user):
    session_active = ChatSession(id="s-active", user_id=test_user.id, title="Active", status="active")
    session_archived = ChatSession(id="s-archived", user_id=test_user.id, title="Archived", status="archived")
    db_session.add_all([session_active, session_archived])
    await db_session.commit()

    resp_all = await client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        params={"status": "all"},
    )
    assert resp_all.json()["pagination"]["total"] == 2

    resp_active = await client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers,
        params={"status": "active"},
    )
    active_data = resp_active.json()
    assert active_data["pagination"]["total"] == 1
    assert active_data["conversations"][0]["status"] == "active"


@pytest.mark.asyncio
async def test_list_sessions_message_count(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "msgcntuser", "email": "msgcnt@test.com", "password": "pass"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"username": "msgcntuser", "password": "pass"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    session_resp = await client.post(
        "/api/v1/chat/sessions",
        headers=headers,
        json={"title": "Count Session"},
    )
    session_id = session_resp.json()["session_id"]
    await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=headers,
        json={"content": "msg1"},
    )
    await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers=headers,
        json={"content": "msg2"},
    )
    response = await client.get(
        "/api/v1/chat/sessions",
        headers=headers,
    )
    convs = response.json()["conversations"]
    target = [c for c in convs if c["id"] == session_id][0]
    assert target["message_count"] == 4


@pytest.mark.asyncio
async def test_list_sessions_unauthorized(client):
    response = await client.get("/api/v1/chat/sessions")
    assert response.status_code == 401
