import unittest
from unittest.mock import patch, MagicMock

from app.services.inference.adapter import (
    LlamaCppInferenceAdapter,
    get_inference_service,
    switch_inference_mode,
    VALID_MODES,
)


class TestLlamaCppInferenceAdapter(unittest.TestCase):
    @patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model")
    def test_adapter_lazy_init(self, mock_load):
        mock_load.return_value = MagicMock()
        adapter = LlamaCppInferenceAdapter(model_path="/tmp/test.gguf")
        self.assertIsNone(adapter._service)

    @patch("app.services.inference.llama_cpp_service.LlamaCppService._load_model")
    def test_adapter_infer_returns_dict(self, mock_load):
        mock_llm = MagicMock()
        mock_llm.return_value = {"choices": [{"text": "test response"}]}
        mock_load.return_value = None
        adapter = LlamaCppInferenceAdapter(model_path="/tmp/test.gguf")
        svc = adapter._ensure_service()
        svc._llm = mock_llm
        import asyncio
        result = asyncio.run(adapter._do_inference("query", ["ctx1"], []))
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("model", result)
        self.assertEqual(result["model"], "llamacpp-gguf")

    def test_adapter_is_available_before_init(self):
        adapter = LlamaCppInferenceAdapter(model_path="/tmp/test.gguf")
        self.assertFalse(adapter.is_available)

    def test_adapter_get_model_info_before_init(self):
        adapter = LlamaCppInferenceAdapter(model_path="/tmp/test.gguf")
        info = adapter.get_model_info()
        self.assertEqual(info["inference_mode"], "llamacpp_gguf")
        self.assertFalse(info["is_available"])


class TestGetInferenceService(unittest.TestCase):
    def setUp(self):
        import app.services.inference.adapter as adapter_mod
        adapter_mod._inference_service_instance = None
        adapter_mod._current_mode = None

    @patch("app.core.config.Settings.__init__", return_value=None)
    def test_mock_mode_returns_mock_service(self, mock_init):
        from app.services.model.mock_inference import MockInferenceService
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.inference_mode = "mock"
            mock_settings.mock_inference = True
            svc = get_inference_service()
            self.assertIsInstance(svc, MockInferenceService)

    def test_valid_modes_contains_expected(self):
        self.assertIn("pytorch_direct", VALID_MODES)
        self.assertIn("llamacpp_gguf", VALID_MODES)
        self.assertIn("mock", VALID_MODES)


class TestSwitchInferenceMode(unittest.TestCase):
    def setUp(self):
        import app.services.inference.adapter as adapter_mod
        adapter_mod._inference_service_instance = None
        adapter_mod._current_mode = None

    @patch("app.core.config.settings")
    def test_switch_to_mock_mode(self, mock_settings):
        mock_settings.inference_mode = "pytorch_direct"
        mock_settings.mock_inference = True
        result = switch_inference_mode("mock")
        self.assertEqual(result["new_mode"], "mock")
        self.assertIn("old_mode", result)

    @patch("app.core.config.settings")
    def test_switch_to_invalid_mode_raises(self, mock_settings):
        with self.assertRaises(ValueError):
            switch_inference_mode("invalid_mode")

    @patch("app.core.config.settings")
    def test_switch_to_llamacpp_without_path_raises(self, mock_settings):
        mock_settings.llamacpp_model_path = None
        mock_settings.mock_inference = True
        with self.assertRaises(ValueError):
            switch_inference_mode("llamacpp_gguf")


if __name__ == "__main__":
    unittest.main()
