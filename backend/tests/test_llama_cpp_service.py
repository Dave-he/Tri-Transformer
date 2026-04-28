import unittest
from unittest.mock import patch, MagicMock

from app.services.inference.llama_cpp_service import LlamaCppService


class TestLlamaCppService(unittest.TestCase):
    def _make_service_with_mock_llm(self):
        mock_llm = MagicMock()
        with patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model"):
            service = LlamaCppService(model_path="/tmp/test.gguf")
            service._llm = mock_llm
        return service, mock_llm

    def test_init_with_model_path(self):
        with patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model"):
            service = LlamaCppService(model_path="/tmp/test.gguf")
            self.assertEqual(service.model_path, "/tmp/test.gguf")

    def test_generate_returns_string(self):
        service, mock_llm = self._make_service_with_mock_llm()
        mock_llm.return_value = {"choices": [{"text": "Hello world"}]}
        result = service.generate("test prompt", max_tokens=64)
        self.assertEqual(result, "Hello world")

    def test_chat_returns_string(self):
        service, mock_llm = self._make_service_with_mock_llm()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hi there"}}]
        }
        result = service.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(result, "Hi there")

    def test_import_error_when_llama_cpp_not_available(self):
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with self.assertRaises(ImportError):
                LlamaCppService(model_path="/tmp/test.gguf")

    def test_n_gpu_layers_configurable(self):
        with patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model"):
            service = LlamaCppService(model_path="/tmp/test.gguf", n_gpu_layers=0)
            self.assertEqual(service.n_gpu_layers, 0)

    def test_is_available_property(self):
        service, _ = self._make_service_with_mock_llm()
        self.assertTrue(service.is_available)

    def test_not_available_when_llm_none(self):
        with patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model"):
            service = LlamaCppService(model_path="/tmp/test.gguf")
            service._llm = None
            self.assertFalse(service.is_available)

    def test_get_model_info(self):
        service, _ = self._make_service_with_mock_llm()
        info = service.get_model_info()
        self.assertEqual(info["inference_mode"], "llamacpp_gguf")
        self.assertTrue(info["is_available"])


if __name__ == "__main__":
    unittest.main()
