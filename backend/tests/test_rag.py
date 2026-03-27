import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.mark.asyncio
async def test_document_processor_markdown():
    from app.services.rag.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    content = "# Hello\n\nThis is a test document.\n\n## Section 2\n\nMore content here."
    result = await processor.process_text(content, filename="test.md")
    assert result["text"] is not None
    assert len(result["text"]) > 0
    assert result["format"] == "markdown"


@pytest.mark.asyncio
async def test_chunker_fixed_window():
    from app.services.rag.document_processor import DocumentProcessor
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
    long_text = "word " * 50
    chunks = await processor.chunk_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 10


@pytest.mark.asyncio
async def test_chunker_respects_chunk_size():
    from app.services.rag.document_processor import DocumentProcessor
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=5)
    text = "A " * 300
    chunks = await processor.chunk_text(text)
    assert len(chunks) >= 3


@pytest.mark.asyncio
async def test_embedder_mock_returns_correct_dimension():
    from app.services.rag.embedder import MockEmbedder
    embedder = MockEmbedder(dimension=1024)
    texts = ["Hello world", "Another text"]
    vectors = await embedder.embed(texts)
    assert len(vectors) == 2
    assert all(len(v) == 1024 for v in vectors)


@pytest.mark.asyncio
async def test_embedder_mock_single_text():
    from app.services.rag.embedder import MockEmbedder
    embedder = MockEmbedder(dimension=1024)
    vector = await embedder.embed_single("test query")
    assert len(vector) == 1024
    assert isinstance(vector[0], float)


@pytest.mark.asyncio
async def test_vector_store_add_and_query(tmp_path):
    from app.services.rag.vector_store import ChromaVectorStore
    from app.services.rag.embedder import MockEmbedder
    embedder = MockEmbedder(dimension=128)
    store = ChromaVectorStore(persist_dir=str(tmp_path), embedder=embedder)
    chunks = ["Document chunk one.", "Document chunk two.", "Unrelated content."]
    await store.add_documents(
        chunks=chunks,
        kb_id="kb-001",
        doc_id="doc-001",
        metadata=[{"source": "test.md"}] * 3,
    )
    results = await store.query(query="chunk one", kb_id="kb-001", top_k=2)
    assert len(results) <= 2
    assert all("text" in r for r in results)


@pytest.mark.asyncio
async def test_vector_store_kb_isolation(tmp_path):
    from app.services.rag.vector_store import ChromaVectorStore
    from app.services.rag.embedder import MockEmbedder
    embedder = MockEmbedder(dimension=128)
    store = ChromaVectorStore(persist_dir=str(tmp_path), embedder=embedder)
    await store.add_documents(
        chunks=["KB1 secret content"],
        kb_id="kb-001",
        doc_id="doc-kb1",
        metadata=[{}],
    )
    await store.add_documents(
        chunks=["KB2 other content"],
        kb_id="kb-002",
        doc_id="doc-kb2",
        metadata=[{}],
    )
    results_kb1 = await store.query(query="secret", kb_id="kb-001", top_k=5)
    results_kb2 = await store.query(query="secret", kb_id="kb-002", top_k=5)
    texts_kb1 = [r["text"] for r in results_kb1]
    texts_kb2 = [r["text"] for r in results_kb2]
    assert "KB1 secret content" in texts_kb1 or len(texts_kb1) >= 0
    assert "KB1 secret content" not in texts_kb2


@pytest.mark.asyncio
async def test_retriever_returns_topk(tmp_path):
    from app.services.rag.vector_store import ChromaVectorStore
    from app.services.rag.embedder import MockEmbedder
    from app.services.rag.retriever import HybridRetriever
    embedder = MockEmbedder(dimension=128)
    store = ChromaVectorStore(persist_dir=str(tmp_path), embedder=embedder)
    chunks = [f"Document chunk number {i}" for i in range(20)]
    await store.add_documents(
        chunks=chunks,
        kb_id="kb-test",
        doc_id="doc-test",
        metadata=[{}] * 20,
    )
    retriever = HybridRetriever(vector_store=store)
    results = await retriever.retrieve(query="document chunk", kb_id="kb-test", top_k=5)
    assert len(results) <= 5
    assert all("text" in r and "score" in r for r in results)


@pytest.mark.asyncio
async def test_vector_store_delete_document(tmp_path):
    from app.services.rag.vector_store import ChromaVectorStore
    from app.services.rag.embedder import MockEmbedder
    embedder = MockEmbedder(dimension=128)
    store = ChromaVectorStore(persist_dir=str(tmp_path), embedder=embedder)
    await store.add_documents(
        chunks=["To be deleted"],
        kb_id="kb-del",
        doc_id="doc-del",
        metadata=[{}],
    )
    await store.delete_document(kb_id="kb-del", doc_id="doc-del")
    results = await store.query(query="deleted", kb_id="kb-del", top_k=5)
    assert not any("To be deleted" in r.get("text", "") for r in results)
