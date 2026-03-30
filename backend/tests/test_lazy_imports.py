def test_document_processor_import_without_fitz():
    from app.services.rag.document_processor import DocumentProcessor
    dp = DocumentProcessor()
    assert dp is not None


def test_vector_store_import_without_chromadb():
    from app.services.rag.vector_store import ChromaVectorStore
    assert ChromaVectorStore is not None


def test_hybrid_retriever_import_without_rank_bm25():
    from app.services.rag.retriever import HybridRetriever
    assert HybridRetriever is not None


def test_document_processor_instantiation():
    from app.services.rag.document_processor import DocumentProcessor
    dp = DocumentProcessor(chunk_size=128, chunk_overlap=10)
    assert dp.chunk_size == 128
    assert dp.chunk_overlap == 10


def test_tri_transformer_inference_service_import():
    from app.services.model.tri_transformer_inference import TriTransformerInferenceService
    from app.services.model.inference_service import InferenceService
    assert issubclass(TriTransformerInferenceService, InferenceService)


def test_tri_transformer_inference_service_has_do_inference():
    from app.services.model.tri_transformer_inference import TriTransformerInferenceService
    assert hasattr(TriTransformerInferenceService, "_do_inference")
