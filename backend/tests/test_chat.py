import pytest


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
    assert "content" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["content"]) > 0


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
    assert "sources" in data


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
