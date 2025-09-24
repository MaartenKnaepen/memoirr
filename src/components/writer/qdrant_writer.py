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
from typing import Any, Dict, List
from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger


@component
class QdrantWriter:
    """Write Documents to Qdrant using configuration from Settings."""

    def __init__(self) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        self._settings = settings
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
        
        self._logger.info(
            "QdrantWriter initialized successfully",
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            embedding_dimension=embedding_dimension,
            component="qdrant_writer"
        )

    @component.output_types(stats=dict)
    def run(self, documents: List[Dict[str, Any]] ) -> dict[str, object]:  # type: ignore[override]
        """Write documents to Qdrant.

        Args:
            documents: list of dicts; each dict should have content, embedding, and optional meta
        """
        with LoggedOperation("document_writing", self._logger, total_documents=len(documents)) as op:
            docs: List[Document] = []
            skipped = 0
            
            for i, d in enumerate(documents):
                try:
                    content = d.get("content")
                    embedding = d.get("embedding")
                    meta = d.get("meta", None)
                    
                    # Validate required fields
                    if not content:
                        self._logger.warning(
                            "Document missing required field",
                            document_index=i,
                            missing_field="content",
                            action="skipped",
                            component="qdrant_writer"
                        )
                        skipped += 1
                        continue
                        
                    if not embedding or not isinstance(embedding, list):
                        self._logger.warning(
                            "Document missing or invalid embedding",
                            document_index=i,
                            missing_field="embedding",
                            embedding_type=type(embedding).__name__ if embedding else "None",
                            action="skipped",
                            component="qdrant_writer"
                        )
                        skipped += 1
                        continue
                    
                    docs.append(Document(content=content, embedding=embedding, meta=meta))
                    
                    self._logger.debug(
                        "Document processed successfully",
                        document_index=i,
                        content_length=len(content),
                        embedding_dimension=len(embedding),
                        has_metadata=meta is not None,
                        component="qdrant_writer"
                    )
                    
                except Exception as e:
                    self._logger.warning(
                        "Failed to process document",
                        document_index=i,
                        error=str(e),
                        error_type=type(e).__name__,
                        action="skipped",
                        component="qdrant_writer"
                    )
                    skipped += 1
                    continue
            
            written_count = 0
            if docs:
                try:
                    self._logger.info(
                        "Writing documents to Qdrant",
                        document_count=len(docs),
                        component="qdrant_writer"
                    )
                    self._store.write_documents(docs)
                    written_count = len(docs)
                    
                    self._logger.info(
                        "Documents written successfully",
                        written_count=written_count,
                        component="qdrant_writer"
                    )
                    
                except Exception as e:
                    self._logger.error(
                        "Failed to write documents to vector store",
                        error=str(e),
                        error_type=type(e).__name__,
                        document_count=len(docs),
                        component="qdrant_writer"
                    )
                    raise
            
            # Add final context and metrics
            op.add_context(
                documents_written=written_count,
                documents_skipped=skipped,
                success_rate=written_count / len(documents) if documents else 0
            )
            
            self._metrics.counter("documents_written_total", written_count, component="qdrant_writer", status="success")
            if skipped > 0:
                self._metrics.counter("documents_skipped_total", skipped, component="qdrant_writer", reason="validation_failed")
            
            return {"stats": {"written": written_count, "skipped": skipped, "total": len(documents)}}

    def clear_collection(self) -> bool:
        """Clear all documents from the Qdrant collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with LoggedOperation("collection_clearing", self._logger) as op:
                self._logger.info(
                    "Clearing Qdrant collection",
                    collection_name=self._settings.qdrant_collection,
                    component="qdrant_writer"
                )
                
                # Delete all documents by using an empty filter (matches all)
                self._store.delete_documents()
                
                op.add_context(collection_cleared=True)
                
                self._logger.info(
                    "Qdrant collection cleared successfully",
                    collection_name=self._settings.qdrant_collection,
                    component="qdrant_writer"
                )
                
                self._metrics.counter("collection_cleared_total", 1, component="qdrant_writer", status="success")
                
                return True
                
        except Exception as e:
            self._logger.error(
                "Failed to clear Qdrant collection",
                collection_name=self._settings.qdrant_collection,
                error=str(e),
                error_type=type(e).__name__,
                component="qdrant_writer"
            )
            
            self._metrics.counter("collection_cleared_total", 1, component="qdrant_writer", status="failed")
            
            return False

    def get_document_count(self) -> int:
        """Get the current number of documents in the collection.
        
        Returns:
            Number of documents in the collection, or -1 if error
        """
        try:
            # Use the document store's count functionality if available
            # This might need adjustment based on the actual QdrantDocumentStore API
            count = self._store.count_documents()
            
            self._logger.debug(
                "Retrieved document count",
                document_count=count,
                component="qdrant_writer"
            )
            
            return count
            
        except Exception as e:
            self._logger.error(
                "Failed to get document count",
                error=str(e),
                error_type=type(e).__name__,
                component="qdrant_writer"
            )
            return -1
