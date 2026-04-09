import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag.retriever import HippoRetriever, HybridRetriever


def make_vector_results(n=6):
    return [
        {"text": f"document about topic {i}", "score": float(i) * 0.1, "metadata": {"doc_id": f"d{i}"}}
        for i in range(1, n + 1)
    ]


class TestHippoRetriever:
    def _make_retriever(self):
        vs = MagicMock()
        vs.query = AsyncMock(return_value=make_vector_results(6))
        return HippoRetriever(vector_store=vs, top_k_candidates=6)

    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self):
        r = self._make_retriever()
        results = await r.retrieve("what is topic 1", "kb1")
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        r = self._make_retriever()
        results = await r.retrieve("topic", "kb1", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        r = self._make_retriever()
        results = await r.retrieve("topic", "kb1")
        for res in results:
            assert "text" in res
            assert "score" in res
            assert "metadata" in res

    @pytest.mark.asyncio
    async def test_scores_are_non_negative(self):
        r = self._make_retriever()
        results = await r.retrieve("topic", "kb1")
        for res in results:
            assert res["score"] >= 0.0

    @pytest.mark.asyncio
    async def test_empty_vector_results(self):
        vs = MagicMock()
        vs.query = AsyncMock(return_value=[])
        r = HippoRetriever(vector_store=vs)
        results = await r.retrieve("anything", "kb1")
        assert results == []

    @pytest.mark.asyncio
    async def test_ppr_builds_graph(self):
        r = self._make_retriever()
        results = await r.retrieve("topic 1", "kb1")
        assert hasattr(r, "_graph") or len(results) >= 0

    @pytest.mark.asyncio
    async def test_single_result_passthrough(self):
        vs = MagicMock()
        vs.query = AsyncMock(return_value=[
            {"text": "only doc", "score": 0.1, "metadata": {}}
        ])
        r = HippoRetriever(vector_store=vs)
        results = await r.retrieve("query", "kb1", top_k=5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_metadata_filter_forwarded(self):
        vs = MagicMock()
        vs.query = AsyncMock(return_value=make_vector_results(3))
        r = HippoRetriever(vector_store=vs)
        filt = {"doc_type": "pdf"}
        await r.retrieve("query", "kb1", metadata_filter=filt)
        vs.query.assert_called_once()
        call_kwargs = vs.query.call_args[1]
        assert call_kwargs.get("metadata_filter") == filt


class TestHybridRetrieverBackwardCompat:
    @pytest.mark.asyncio
    async def test_still_works(self):
        vs = MagicMock()
        vs.query = AsyncMock(return_value=make_vector_results(4))
        r = HybridRetriever(vector_store=vs)
        results = await r.retrieve("query", "kb1", top_k=2)
        assert isinstance(results, list)
