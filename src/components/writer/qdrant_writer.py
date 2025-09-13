"""Haystack component that writes Documents with embeddings to Qdrant.

Reads Qdrant configuration from Settings/.env:
- QDRANT_URL: base URL or ":memory:" for in-memory
- QDRANT_COLLECTION: collection name
- QDRANT_RECREATE_INDEX: whether to recreate the collection at startup
- QDRANT_RETURN_EMBEDDING: whether store returns embeddings
- QDRANT_WAIT_RESULT: whether to wait for API result

Input contract (JSON-serializable):
- documents: list of dicts with keys {"content": str, "embedding": list[float], "meta": dict (optional)}

Outputs:
- stats: dict with counts written
"""

from haystack import component
from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.core.config import get_settings


@component
class QdrantWriter:
    """Write Documents to Qdrant using configuration from Settings."""

    def __init__(self) -> None:
        settings = get_settings()
        kwargs = dict(
            url=settings.qdrant_url,
            index=settings.qdrant_collection,
            recreate_index=settings.qdrant_recreate_index,
            return_embedding=settings.qdrant_return_embedding,
            wait_result_from_api=settings.qdrant_wait_result,
        )
        embedding_dimension = getattr(settings, "embedding_dimension", None)
        if embedding_dimension is not None:
            kwargs["embedding_dim"] = embedding_dimension
        self._store = QdrantDocumentStore(**kwargs)

    @component.output_types(stats=dict)
    def run(self, documents: list[dict]) -> dict[str, object]:  # type: ignore[override]
        """Write documents to Qdrant.

        Args:
            documents: list of dicts; each dict should have content, embedding, and optional meta
        """
        docs: list[Document] = []
        for d in documents:
            content = d.get("content")
            embedding = d.get("embedding")
            meta = d.get("meta", None)
            docs.append(Document(content=content, embedding=embedding, meta=meta))
        if docs:
            self._store.write_documents(docs)
        return {"stats": {"written": len(docs)}}
