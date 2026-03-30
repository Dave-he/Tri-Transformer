import requests
from unittest.mock import MagicMock, patch

import pytest

from app.services.model.ollama_client import OllamaClient


class TestOllamaClient:
    def _mock_generate_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "hello from gemma", "done": True}
        return mock_resp

    def _mock_tags_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "gemma3:4b"},
                {"name": "gemma3:1b"},
            ]
        }
        return mock_resp

    @patch("app.services.model.ollama_client.requests.post")
    def test_generate_returns_string(self, mock_post):
        mock_post.return_value = self._mock_generate_response()
        client = OllamaClient()
        result = client.generate("gemma3:4b", "hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("app.services.model.ollama_client.requests.post")
    def test_generate_returns_response_text(self, mock_post):
        mock_post.return_value = self._mock_generate_response()
        client = OllamaClient()
        result = client.generate("gemma3:4b", "hello")
        assert result == "hello from gemma"

    @patch("app.services.model.ollama_client.requests.get")
    def test_list_models_returns_list(self, mock_get):
        mock_get.return_value = self._mock_tags_response()
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)
        assert "gemma3:4b" in models

    @patch("app.services.model.ollama_client.requests.get")
    def test_is_available_true_when_reachable(self, mock_get):
        mock_get.return_value = self._mock_tags_response()
        client = OllamaClient()
        assert client.is_available() is True

    @patch("app.services.model.ollama_client.requests.get")
    def test_is_available_false_on_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        client = OllamaClient()
        assert client.is_available() is False

    @patch("app.services.model.ollama_client.requests.post")
    def test_generate_calls_correct_endpoint(self, mock_post):
        mock_post.return_value = self._mock_generate_response()
        client = OllamaClient(base_url="http://localhost:11434")
        client.generate("gemma3:4b", "test prompt")
        call_url = mock_post.call_args[0][0]
        assert "/api/generate" in call_url
