import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_inference_mock_mode(client, auth_headers):
    response = await client.post(
        "/api/v1/model/inference",
        headers=auth_headers,
        json={
            "query": "What is Tri-Transformer?",
            "context": ["Tri-Transformer is a three-branch architecture."],
            "history": [],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "confidence" in data
    assert isinstance(data["text"], str)
    assert len(data["text"]) > 0
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_inference_missing_query(client, auth_headers):
    response = await client.post(
        "/api/v1/model/inference",
        headers=auth_headers,
        json={"context": [], "history": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_inference_empty_context(client, auth_headers):
    response = await client.post(
        "/api/v1/model/inference",
        headers=auth_headers,
        json={"query": "What is RAG?", "context": [], "history": []},
    )
    assert response.status_code == 200
    assert "text" in response.json()


@pytest.mark.asyncio
async def test_inference_with_history(client, auth_headers):
    response = await client.post(
        "/api/v1/model/inference",
        headers=auth_headers,
        json={
            "query": "Tell me more.",
            "context": ["RAG stands for Retrieval Augmented Generation."],
            "history": [
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG is a technique."},
            ],
        },
    )
    assert response.status_code == 200
    assert "text" in response.json()


@pytest.mark.asyncio
async def test_inference_unauthorized(client):
    response = await client.post(
        "/api/v1/model/inference",
        json={"query": "test", "context": [], "history": []},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_inference_retry_on_failure():
    from app.services.model.inference_service import InferenceService
    call_count = 0

    class FailingService(InferenceService):
        async def _do_inference(self, query, context, history):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return {"text": "Success after retry", "confidence": 0.9}

    service = FailingService()
    result = await service.infer(query="test", context=[], history=[])
    assert result["text"] == "Success after retry"
    assert call_count == 3


@pytest.mark.asyncio
async def test_inference_max_retries_exceeded():
    from app.services.model.inference_service import InferenceService, InferenceError

    class AlwaysFailingService(InferenceService):
        async def _do_inference(self, query, context, history):
            raise RuntimeError("Always fails")

    service = AlwaysFailingService()
    with pytest.raises(InferenceError):
        await service.infer(query="test", context=[], history=[])
