import hashlib
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
        self._cache: dict[str, list[float]] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append([])
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self._model.encode(uncached_texts, normalize_embeddings=True).tolist()
            for idx, text, vec in zip(uncached_indices, uncached_texts, embeddings):
                self._cache[self._cache_key(text)] = vec
                results[idx] = vec

        return results

    async def embed_single(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        result = await self.embed([text])
        return result[0]


def get_embedder() -> BaseEmbedder:
    if settings.mock_inference:
        return MockEmbedder()
    return BGEEmbedder()
