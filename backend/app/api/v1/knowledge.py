import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.config import settings
from app.dependencies import get_current_user
from app.models.user import User
from app.models.document import Document
from app.schemas.knowledge import (
    DocumentResponse, DocumentUploadResponse, DocumentListResponse,
    SearchResponse, SearchResultItem, DocumentStatusResponse,
    SearchRequest, DocumentDeleteResponse,
)
from app.services.rag.document_processor import DocumentProcessor
from app.services.rag.embedder import get_embedder
from app.services.rag.vector_store import ChromaVectorStore
from app.services.rag.retriever import HybridRetriever

router = APIRouter()

logger = logging.getLogger("tri-transformer")

SUPPORTED_EXTENSIONS = {"pdf", "md", "markdown", "txt", "docx"}

ALLOWED_MIMES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def get_vector_store() -> ChromaVectorStore:
    embedder = get_embedder()
    return ChromaVectorStore(embedder=embedder)


async def _process_document(
    doc_id: str,
    file_content: bytes,
    filename: str,
    kb_id: str,
    db_url: str,
):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    engine = create_async_engine(db_url)
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        processed = await processor.process_file(file_content, filename)
        chunks = await processor.chunk_text(processed["text"])

        embedder = get_embedder()
        store = ChromaVectorStore(embedder=embedder)
        await store.add_documents(
            chunks=chunks,
            kb_id=kb_id,
            doc_id=doc_id,
            metadata=[{"filename": filename}] * len(chunks),
        )

        async with sm() as session:
            result = await session.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "ready"
                doc.chunk_count = len(chunks)
                await session.commit()
    except Exception as exc:
        logger.exception("Document processing failed for doc_id=%s: %s", doc_id, exc)
        try:
            async with sm() as session:
                result = await session.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    await session.commit()
        except Exception as inner_exc:
            logger.exception("Failed to update document status to failed: %s", inner_exc)
    finally:
        try:
            await engine.dispose()
        except Exception as dispose_exc:
            logger.warning("Failed to dispose engine: %s", dispose_exc)


@router.post("/documents", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file format: {ext}")

    file_content = await file.read()

    try:
        import magic
        detected_mime = magic.Magic(mime=True).from_buffer(file_content)
        if detected_mime not in ALLOWED_MIMES:
            raise HTTPException(status_code=422, detail=f"不支持的文件类型: {detected_mime}")
    except ImportError:
        logger.warning("python-magic 未安装，跳过 MIME 类型校验")

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=current_user.id,
        kb_id=current_user.kb_id,
        filename=file.filename,
        status="processing",
    )
    db.add(doc)
    await db.commit()

    background_tasks.add_task(
        _process_document,
        doc_id=doc_id,
        file_content=file_content,
        filename=file.filename,
        kb_id=current_user.kb_id,
        db_url=settings.database_url,
    )

    return DocumentUploadResponse(document_id=doc_id, status="processing")


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.kb_id == current_user.kb_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    progress = 100 if doc.status == "ready" else 0 if doc.status == "processing" else -1
    return DocumentStatusResponse(
        document_id=doc.id,
        status=doc.status,
        progress=progress,
        chunk_count=doc.chunk_count,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(Document.kb_id == current_user.kb_id)
    )
    docs = result.scalars().all()
    documents = [
        DocumentResponse(
            document_id=d.id,
            filename=d.filename,
            type=d.filename.rsplit(".", 1)[-1].lower() if "." in d.filename else "",
            size=d.file_size if hasattr(d, "file_size") else 0,
            status=d.status,
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in docs
    ]
    return DocumentListResponse(documents=documents)


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.kb_id == current_user.kb_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    store = get_vector_store()
    await store.delete_document(kb_id=current_user.kb_id, doc_id=document_id)

    await db.delete(doc)
    await db.commit()
    return DocumentDeleteResponse(message="Document deleted")


@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    payload: SearchRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        store = get_vector_store()
        retriever = HybridRetriever(vector_store=store)
        results = await retriever.retrieve(
            query=payload.query,
            kb_id=current_user.kb_id,
            top_k=payload.top_k,
        )
        if settings.use_reranker:
            from app.services.rag.reranker import BGEReranker
            reranker = BGEReranker()
            results = await reranker.rerank(
                query=payload.query, results=results, top_k=payload.top_k
            )
    except Exception as exc:
        logger.exception("Knowledge search failed for kb_id=%s query=%r: %s", current_user.kb_id, payload.query, exc)
        results = []
    return SearchResponse(
        results=[SearchResultItem(**r) for r in results],
        total=len(results),
    )
