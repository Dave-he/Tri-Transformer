import random
from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        pass


class MockEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [random.gauss(0, 1) for _ in range(self.dimension)]
            for _ in texts
        ]

    async def embed_single(self, text: str) -> list[float]:
        return [random.gauss(0, 1) for _ in range(self.dimension)]


class BGEEmbedder(BaseEmbedder):
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.embedding_model_path
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


def get_embedder() -> BaseEmbedder:
    if settings.mock_inference:
        return MockEmbedder()
    return BGEEmbedder()
