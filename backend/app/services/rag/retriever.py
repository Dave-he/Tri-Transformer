from typing import Optional

import numpy as np

from app.services.rag.vector_store import ChromaVectorStore


class HippoRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        top_k_candidates: int = 6,
    ):
        self.vector_store = vector_store
        self.top_k_candidates = top_k_candidates

    async def retrieve(
        self,
        query: str,
        kb_id: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        candidates = await self.vector_store.query(
            query=query,
            kb_id=kb_id,
            top_k=self.top_k_candidates,
            metadata_filter=metadata_filter,
        )
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        query_tokens = set(query.lower().split())
        ppr_scores = self._run_ppr(candidates, query_tokens)

        for i, score in enumerate(ppr_scores):
            candidates[i]["score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _build_ppr_graph(self, candidates: list[dict]) -> np.ndarray:
        n = len(candidates)
        doc_tokens = [set(c["text"].lower().split()) for c in candidates]
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                overlap = len(doc_tokens[i] & doc_tokens[j])
                if overlap > 0:
                    adj[i][j] = overlap / max(len(doc_tokens[j]), 1)
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        adj = adj / row_sums
        return adj

    def _run_ppr(
        self,
        candidates: list[dict],
        query_tokens: set,
        damping: float = 0.85,
        max_iter: int = 10,
    ) -> np.ndarray:
        n = len(candidates)
        doc_tokens = [set(c["text"].lower().split()) for c in candidates]
        personalization = np.zeros(n, dtype=np.float64)
        for i in range(n):
            overlap = len(query_tokens & doc_tokens[i])
            personalization[i] = overlap / max(len(query_tokens), 1)
        p_sum = personalization.sum()
        if p_sum > 0:
            personalization = personalization / p_sum
        else:
            personalization = np.ones(n, dtype=np.float64) / n

        adj = self._build_ppr_graph(candidates)
        scores = personalization.copy()
        for _ in range(max_iter):
            scores = damping * adj.T @ scores + (1 - damping) * personalization
        return scores


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
        from rank_bm25 import BM25Okapi
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
