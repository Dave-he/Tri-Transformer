from typing import Optional

from app.core.config import settings


class BGEReranker:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.reranker_model_path
        self._model = None
        self._mock = settings.mock_inference

    def _load_model(self):
        if self._model is None and not self._mock:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_path)

    async def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not results:
            return []
        if self._mock:
            import random
            reranked = [
                {**r, "rerank_score": random.random()} for r in results
            ]
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        self._load_model()
        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)
        reranked = [
            {**results[i], "rerank_score": float(scores[i])} for i in range(len(results))
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
