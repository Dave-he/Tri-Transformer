import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_register_success(client):
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "newuser", "email": "new@example.com", "password": "password123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "user_id" in data
    assert data["username"] == "newuser"


@pytest.mark.asyncio
async def test_register_duplicate_username(client):
    payload = {"username": "dupuser", "email": "dup@example.com", "password": "password123"}
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "dupuser", "email": "other@example.com", "password": "password123"},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    payload = {"username": "user1", "email": "same@example.com", "password": "password123"}
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "user2", "email": "same@example.com", "password": "password123"},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "loginuser", "email": "login@example.com", "password": "testpass"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"username": "loginuser", "password": "testpass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post(
        "/api/v1/auth/register",
        json={"username": "authuser", "email": "auth@example.com", "password": "correct"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"username": "authuser", "password": "wrongpassword"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    response = await client.post(
        "/api/v1/auth/login",
        json={"username": "ghost", "password": "password"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_without_token(client):
    response = await client.get("/api/v1/knowledge/documents")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_with_invalid_token(client):
    response = await client.get(
        "/api/v1/knowledge/documents",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert response.status_code == 401
