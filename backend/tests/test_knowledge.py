import pytest
import io
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_upload_document_success(client, auth_headers):
    file_content = b"# Test Document\n\nThis is test content for the knowledge base."
    response = await client.post(
        "/api/v1/knowledge/documents",
        headers=auth_headers,
        files={"file": ("test.md", io.BytesIO(file_content), "text/markdown")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "processing"


@pytest.mark.asyncio
async def test_upload_document_unsupported_format(client, auth_headers):
    response = await client.post(
        "/api/v1/knowledge/documents",
        headers=auth_headers,
        files={"file": ("test.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_documents_returns_own_docs(client, auth_headers):
    file_content = b"# List Test\n\nContent."
    await client.post(
        "/api/v1/knowledge/documents",
        headers=auth_headers,
        files={"file": ("list_test.md", io.BytesIO(file_content), "text/markdown")},
    )
    response = await client.get("/api/v1/knowledge/documents", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_list_documents_kb_isolation(client):
    file_content = b"# Isolation Test\n\nSecret content."
    user1_reg = await client.post(
        "/api/v1/auth/register",
        json={"username": "user_a", "email": "usera@test.com", "password": "pass123"},
    )
    user1_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "user_a", "password": "pass123"},
    )
    token1 = user1_login.json()["access_token"]
    headers1 = {"Authorization": f"Bearer {token1}"}

    user2_reg = await client.post(
        "/api/v1/auth/register",
        json={"username": "user_b", "email": "userb@test.com", "password": "pass123"},
    )
    user2_login = await client.post(
        "/api/v1/auth/login",
        json={"username": "user_b", "password": "pass123"},
    )
    token2 = user2_login.json()["access_token"]
    headers2 = {"Authorization": f"Bearer {token2}"}

    await client.post(
        "/api/v1/knowledge/documents",
        headers=headers1,
        files={"file": ("secret.md", io.BytesIO(file_content), "text/markdown")},
    )

    docs_user2 = await client.get("/api/v1/knowledge/documents", headers=headers2)
    assert docs_user2.status_code == 200
    doc_names = [d["filename"] for d in docs_user2.json()["documents"]]
    assert "secret.md" not in doc_names


@pytest.mark.asyncio
async def test_delete_document(client, auth_headers):
    file_content = b"# To Delete\n\nThis will be deleted."
    upload_resp = await client.post(
        "/api/v1/knowledge/documents",
        headers=auth_headers,
        files={"file": ("todelete.md", io.BytesIO(file_content), "text/markdown")},
    )
    doc_id = upload_resp.json()["document_id"]
    delete_resp = await client.delete(
        f"/api/v1/knowledge/documents/{doc_id}",
        headers=auth_headers,
    )
    assert delete_resp.status_code == 200
    assert "message" in delete_resp.json()


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client, auth_headers):
    response = await client.delete(
        "/api/v1/knowledge/documents/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_knowledge(client, auth_headers):
    response = await client.post(
        "/api/v1/knowledge/search",
        json={"query": "test content", "top_k": 5},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_knowledge_unauthorized(client):
    response = await client.get("/api/v1/knowledge/documents")
    assert response.status_code == 401
