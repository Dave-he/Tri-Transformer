import pytest
from unittest.mock import AsyncMock, patch


class TestFactChecker:
    def test_check_identical_text_high_score(self):
        from app.services.model.fact_checker import FactChecker
        checker = FactChecker()
        result = checker.check("The capital of France is Paris.", ["The capital of France is Paris."])
        assert result.score >= 0.9

    def test_check_different_text_low_score(self):
        from app.services.model.fact_checker import FactChecker
        checker = FactChecker()
        result = checker.check(
            "The sky is green and the grass is blue.",
            ["The capital of France is Paris."]
        )
        assert result.score < 0.3

    def test_check_result_structure(self):
        from app.services.model.fact_checker import FactChecker, FactCheckResult
        checker = FactChecker()
        result = checker.check("test text", ["context"])
        assert isinstance(result, FactCheckResult)
        assert hasattr(result, "score")
        assert hasattr(result, "hallucination_detected")
        assert isinstance(result.score, float)
        assert isinstance(result.hallucination_detected, bool)

    def test_hallucination_threshold(self):
        from app.services.model.fact_checker import FactChecker
        checker = FactChecker(threshold=0.5)
        result = checker.check("xyz unrelated text", ["completely different context"])
        assert result.hallucination_detected is True


class TestChatHallucinationField:
    @pytest.mark.asyncio
    async def test_chat_message_response_has_hallucination_field(self, client, auth_headers, test_user):
        from app.schemas.chat import MessageResponse
        assert "hallucination_detected" in MessageResponse.model_fields

    @pytest.mark.asyncio
    async def test_send_message_includes_hallucination_field(self, client, auth_headers, test_user):
        with patch(
            "app.services.chat.chat_service.ChromaVectorStore",
            return_value=AsyncMock(),
        ), patch(
            "app.services.chat.chat_service.HybridRetriever",
            return_value=AsyncMock(retrieve=AsyncMock(return_value=[])),
        ):
            response = await client.post(
                "/api/v1/chat/sessions",
                headers=auth_headers,
                json={"title": "Test Session"},
            )
            assert response.status_code in (200, 201)
            session_id = response.json()["session_id"]

            msg_response = await client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers,
                json={"content": "Hello, what is the capital of France?"},
            )
            assert msg_response.status_code == 200
            data = msg_response.json()
            assert "hallucination_detected" in data
