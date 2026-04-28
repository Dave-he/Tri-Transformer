import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LlamaCppService:
    """
    llama.cpp inference service wrapper using llama-cpp-python bindings.

    This service provides an alternative lightweight inference path using GGUF quantized models.
    It only supports the O-Transformer Streaming Decoder branch (single causal LM),
    not the full Tri-Transformer I/C/O three-branch architecture.

    For full Tri-Transformer inference with hallucination detection and real-time control,
    use the pytorch_direct inference mode (InferenceService).
    """

    def __init__(self, model_path: str, n_gpu_layers: int = 0):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._load_model(model_path, n_gpu_layers)

    def _load_model(self, model_path: str, n_gpu_layers: int):
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False,
            )
            logger.info("Loaded GGUF model: %s (n_gpu_layers=%d)", model_path, n_gpu_layers)
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Install it via: "
                "pip install llama-cpp-python or CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self._llm is None:
            raise RuntimeError("Model not loaded")
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["text"]

    def chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.7) -> str:
        if self._llm is None:
            raise RuntimeError("Model not loaded")
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["message"]["content"]

    @property
    def is_available(self) -> bool:
        return self._llm is not None

    def get_model_info(self) -> dict:
        return {
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "inference_mode": "llamacpp_gguf",
            "is_available": self.is_available,
        }
