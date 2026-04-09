import pytest
import json
import tempfile
import os

from app.services.rag.graph_index import GraphRAGIndex


SAMPLE_DOCS = [
    {"id": "d1", "text": "Transformer architecture uses self-attention mechanisms for NLP tasks."},
    {"id": "d2", "text": "BERT and GPT are popular transformer models for language understanding."},
    {"id": "d3", "text": "Attention mechanism computes weighted sums over token representations."},
    {"id": "d4", "text": "Knowledge graphs represent entities and relations as triples."},
    {"id": "d5", "text": "Graph neural networks propagate information along graph edges."},
]


class TestGraphRAGIndex:
    def _make_index(self, persist_dir=None):
        return GraphRAGIndex(persist_dir=persist_dir)

    def test_add_documents_and_query(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.global_query("transformer attention")
        assert isinstance(results, list)

    def test_global_query_returns_summaries(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.global_query("transformer architecture")
        for r in results:
            assert "summary" in r or "text" in r

    def test_communities_built(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        assert idx.community_count() >= 1

    def test_empty_query_returns_empty(self):
        idx = self._make_index()
        results = idx.global_query("anything")
        assert results == []

    def test_add_then_delete(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        idx.remove_document("d1")
        count_before = idx.doc_count()
        idx.add_documents([{"id": "d99", "text": "new document about graphs"}])
        assert idx.doc_count() == count_before + 1

    def test_doc_count(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        assert idx.doc_count() == len(SAMPLE_DOCS)

    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = GraphRAGIndex(persist_dir=tmpdir)
            idx.add_documents(SAMPLE_DOCS)
            idx.save()
            idx2 = GraphRAGIndex(persist_dir=tmpdir)
            idx2.load()
            assert idx2.doc_count() == len(SAMPLE_DOCS)

    def test_query_score_ordering(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.global_query("transformer self-attention NLP")
        if len(results) >= 2:
            scores = [r.get("score", 0) for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_update_existing_doc(self):
        idx = self._make_index()
        idx.add_documents(SAMPLE_DOCS)
        updated = [{"id": "d1", "text": "Updated document about transformers and attention heads."}]
        idx.add_documents(updated)
        assert idx.doc_count() == len(SAMPLE_DOCS)
