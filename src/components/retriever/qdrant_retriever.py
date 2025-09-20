"""Haystack component for retrieving similar documents from Qdrant vector store.

This component accepts a query text, embeds it using the configured embedding model,
and retrieves the most similar documents from the Qdrant collection.

It follows Haystack's custom component requirements:
- @component decorator
- run() method returning a dict matching @component.output_types

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import List, Dict, Any, Optional

from haystack import component
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval import orchestrate_retrieval


@component
class QdrantRetriever:
    """Haystack component for retrieving similar documents from Qdrant.

    This component handles the complete retrieval pipeline:
    1. Embeds the input query using the configured embedding model
    2. Searches the Qdrant vector store for similar documents
    3. Returns ranked documents with similarity scores and metadata

    Args:
        top_k: Maximum number of documents to retrieve.
        score_threshold: Minimum similarity score threshold (0.0-1.0).
        return_embedding: Whether to include embeddings in returned documents.
        filters: Optional metadata filters to apply during search.
    """

    def __init__(
        self,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        return_embedding: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)

        # Resolve configuration from args or .env
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
        self.return_embedding = return_embedding if return_embedding is not None else settings.retrieval_return_embedding
        self.filters = filters or {}

        # Initialize Qdrant document store with same configuration as writer
        self._document_store = QdrantDocumentStore(
            url=settings.qdrant_url,
            index=settings.qdrant_collection,
            return_embedding=self.return_embedding,
            wait_result_from_api=settings.qdrant_wait_result,
        )

        self._logger.info(
            "QdrantRetriever initialized successfully",
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            return_embedding=self.return_embedding,
            component="qdrant_retriever"
        )

    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, object]:  # type: ignore[override]
        """Retrieve similar documents from Qdrant for the given query.

        Args:
            query: The search query text to find similar documents for.
            top_k: Override the default number of documents to retrieve.
            filters: Override the default metadata filters for this search.

        Returns:
            Dict with:
            - documents: List of Document objects ranked by similarity score
        """
        with LoggedOperation("qdrant_retrieval", self._logger, query_length=len(query)) as op:
            # Use provided parameters or fall back to instance defaults
            effective_top_k = top_k if top_k is not None else self.top_k
            effective_filters = filters if filters is not None else self.filters

            self._logger.info(
                "Starting document retrieval",
                query_preview=query[:100] + "..." if len(query) > 100 else query,
                top_k=effective_top_k,
                score_threshold=self.score_threshold,
                has_filters=bool(effective_filters),
                component="qdrant_retriever"
            )

            # Delegate to orchestration function
            documents = orchestrate_retrieval(
                query=query,
                document_store=self._document_store,
                top_k=effective_top_k,
                score_threshold=self.score_threshold,
                filters=effective_filters,
            )

            # Add context and metrics
            op.add_context(
                retrieved_documents=len(documents),
                top_k_requested=effective_top_k,
                query_length=len(query),
                avg_score=sum(doc.score for doc in documents if doc.score) / len(documents) if documents else 0.0
            )

            self._metrics.counter("retrieval_queries_total", 1, component="qdrant_retriever")
            self._metrics.counter("documents_retrieved_total", len(documents), component="qdrant_retriever")
            self._metrics.histogram("retrieval_count", len(documents), component="qdrant_retriever")

            if documents:
                self._metrics.histogram("avg_retrieval_score", sum(doc.score for doc in documents if doc.score) / len(documents), component="qdrant_retriever")

            self._logger.info(
                "Document retrieval completed",
                retrieved_documents=len(documents),
                top_k_requested=effective_top_k,
                avg_score=sum(doc.score for doc in documents if doc.score) / len(documents) if documents else 0.0,
                component="qdrant_retriever"
            )

            return {"documents": documents}