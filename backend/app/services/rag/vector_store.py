from typing import Optional

from app.services.rag.embedder import BaseEmbedder
from app.core.config import settings


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
    ):
        import chromadb
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = (
            chromadb.Client()
            if persist_dir == ":memory:"
            else chromadb.PersistentClient(path=self.persist_dir)
        )
        self.embedder = embedder

    def _get_collection(self, kb_id: str):
        collection_name = f"kb_{kb_id.replace('-', '_')}"
        return self._client.get_or_create_collection(name=collection_name)

    async def add_documents(
        self,
        chunks: list[str],
        kb_id: str,
        doc_id: str,
        metadata: Optional[list[dict]] = None,
    ):
        collection = self._get_collection(kb_id)
        vectors = await self.embedder.embed(chunks)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metas = metadata or [{} for _ in chunks]
        for i, m in enumerate(metas):
            metas[i] = {**m, "doc_id": doc_id, "kb_id": kb_id}
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=metas,
        )

    async def query(
        self,
        query: str,
        kb_id: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        collection = self._get_collection(kb_id)
        query_vector = await self.embedder.embed_single(query)
        where = metadata_filter or {}
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, collection.count())),
            where=where if where else None,
        )
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append(
                    {
                        "text": doc,
                        "score": (
                            float(results["distances"][0][i])
                            if results.get("distances") else 0.0
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if results.get("metadatas") else {}
                        ),
                    }
                )
        return output

    async def delete_document(self, kb_id: str, doc_id: str):
        collection = self._get_collection(kb_id)
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])

    async def get_doc_count(self, kb_id: str) -> int:
        collection = self._get_collection(kb_id)
        return collection.count()
