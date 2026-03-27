from typing import Optional

from rank_bm25 import BM25Okapi

from app.services.rag.vector_store import ChromaVectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    async def retrieve(
        self,
        query: str,
        kb_id: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        vector_results = await self.vector_store.query(
            query=query,
            kb_id=kb_id,
            top_k=top_k * 2,
            metadata_filter=metadata_filter,
        )
        if not vector_results:
            return []

        texts = [r["text"] for r in vector_results]
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        max_dist = max((r["score"] for r in vector_results), default=1.0) or 1.0

        combined = []
        for i, r in enumerate(vector_results):
            vec_score = 1.0 - (r["score"] / max_dist) if max_dist > 0 else 0.5
            bm25_norm = float(bm25_scores[i]) / max_bm25 if max_bm25 > 0 else 0.0
            combined_score = self.vector_weight * vec_score + self.bm25_weight * bm25_norm
            combined.append({**r, "score": combined_score})

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
