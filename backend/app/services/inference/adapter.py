import asyncio
import logging
from typing import Optional

from app.services.model.inference_service import InferenceService

logger = logging.getLogger(__name__)

VALID_MODES = {"pytorch_direct", "llamacpp_gguf", "mock"}


class LlamaCppInferenceAdapter(InferenceService):

    def __init__(self, model_path: str, n_gpu_layers: int = 0):
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._service: Optional[object] = None

    def _ensure_service(self):
        if self._service is None:
            from app.services.inference.llama_cpp_service import LlamaCppService
            self._service = LlamaCppService(
                model_path=self._model_path,
                n_gpu_layers=self._n_gpu_layers,
            )
        return self._service

    async def _do_inference(
        self,
        query: str,
        context: list[str],
        history: list[dict],
    ) -> dict:
        svc = self._ensure_service()
        prompt_parts = []
        if context:
            prompt_parts.append("Context:\n" + "\n".join(f"- {c}" for c in context))
        if history:
            for h in history[-6:]:
                role = h.get("role", "user")
                content = h.get("content", "")
                prompt_parts.append(f"{role}: {content}")
        prompt_parts.append(f"user: {query}")
        prompt_parts.append("assistant:")
        prompt = "\n".join(prompt_parts)
        text = await asyncio.to_thread(svc.generate, prompt, 256, 0.7)
        return {
            "text": text,
            "confidence": 0.70,
            "model": "llamacpp-gguf",
        }

    @property
    def is_available(self) -> bool:
        if self._service is None:
            return False
        return self._service.is_available

    def get_model_info(self) -> dict:
        if self._service is None:
            return {
                "model_path": self._model_path,
                "n_gpu_layers": self._n_gpu_layers,
                "inference_mode": "llamacpp_gguf",
                "is_available": False,
            }
        return self._service.get_model_info()


_inference_service_instance: Optional[InferenceService] = None
_current_mode: Optional[str] = None


def get_inference_service() -> InferenceService:
    global _inference_service_instance, _current_mode
    from app.core.config import settings

    mode = settings.inference_mode
    if _inference_service_instance is not None and _current_mode == mode:
        return _inference_service_instance

    if mode == "mock" or (mode == "pytorch_direct" and settings.mock_inference):
        from app.services.model.mock_inference import MockInferenceService
        svc = MockInferenceService()
    elif mode == "llamacpp_gguf":
        if not settings.llamacpp_model_path:
            raise ValueError(
                "llamacpp_gguf mode requires llamacpp_model_path setting"
            )
        svc = LlamaCppInferenceAdapter(
            model_path=settings.llamacpp_model_path,
            n_gpu_layers=0,
        )
    else:
        from app.services.model.tri_transformer_inference import TriTransformerInferenceService
        svc = TriTransformerInferenceService()

    _inference_service_instance = svc
    _current_mode = mode
    logger.info("Inference service initialized: mode=%s", mode)
    return svc


def switch_inference_mode(new_mode: str) -> dict:
    global _inference_service_instance, _current_mode
    from app.core.config import settings

    if new_mode not in VALID_MODES:
        raise ValueError(f"Invalid inference mode: {new_mode}. Valid modes: {VALID_MODES}")

    if new_mode == "llamacpp_gguf" and not settings.llamacpp_model_path:
        raise ValueError("llamacpp_gguf mode requires llamacpp_model_path setting")

    old_mode = _current_mode or settings.inference_mode
    _inference_service_instance = None
    _current_mode = None
    settings.inference_mode = new_mode

    try:
        svc = get_inference_service()
    except Exception:
        settings.inference_mode = old_mode
        _inference_service_instance = None
        _current_mode = None
        raise

    return {
        "old_mode": old_mode,
        "new_mode": new_mode,
        "service_available": True,
    }
