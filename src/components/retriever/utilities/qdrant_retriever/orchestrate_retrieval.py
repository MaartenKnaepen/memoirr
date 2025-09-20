"""Orchestration function for Qdrant document retrieval.

This module handles the complete retrieval pipeline:
1. Embeds the query text using the configured embedding model
2. Searches Qdrant for similar document embeddings
3. Filters results by score threshold
4. Returns ranked documents with metadata

Follows Memoirr patterns: pure functions, proper error handling, comprehensive logging.
"""

from typing import List, Dict, Any, Optional

from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.embedder.text_embedder import TextEmbedder


def orchestrate_retrieval(
    query: str,
    document_store: QdrantDocumentStore,
    top_k: int,
    score_threshold: float,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Orchestrate the complete retrieval process from query to ranked documents.

    Args:
        query: The search query text.
        document_store: Initialized Qdrant document store.
        top_k: Maximum number of documents to retrieve.
        score_threshold: Minimum similarity score threshold (0.0-1.0).
        filters: Optional metadata filters to apply during search.

    Returns:
        List of Document objects ranked by similarity score, filtered by threshold.

    Raises:
        ValueError: If query is empty or invalid parameters.
        RuntimeError: If embedding or retrieval fails.
    """
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)

    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only")
    
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be between 0.0 and 1.0")

    with LoggedOperation("retrieval_orchestration", logger, query_length=len(query)) as op:
        try:
            # Step 1: Embed the query
            logger.debug(
                "Starting query embedding",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                component="retrieval_orchestrator"
            )

            embedder = TextEmbedder()
            embedding_result = embedder.run(text=[query])
            query_embedding = embedding_result["embedding"][0]  # Get first (and only) embedding

            logger.debug(
                "Query embedding completed",
                embedding_dimension=len(query_embedding),
                component="retrieval_orchestrator"
            )

            # Step 2: Search Qdrant for similar documents using QdrantEmbeddingRetriever
            logger.debug(
                "Starting vector search",
                top_k=top_k,
                score_threshold=score_threshold,
                has_filters=bool(filters),
                component="retrieval_orchestrator"
            )

            # Initialize the retriever
            retriever = QdrantEmbeddingRetriever(
                document_store=document_store,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters
            )

            # Perform the search
            search_result = retriever.run(query_embedding=query_embedding)
            search_results = search_result["documents"]

            logger.debug(
                "Vector search completed",
                raw_results_count=len(search_results),
                component="retrieval_orchestrator"
            )

            # Step 3: Add retrieval metadata (filtering is already done by the retriever)
            for i, doc in enumerate(search_results):
                if doc.meta is None:
                    doc.meta = {}
                
                doc.meta.update({
                    "retrieval_rank": i + 1,
                    "retrieval_query": query[:100] + "..." if len(query) > 100 else query,
                    "retrieval_score": doc.score,
                })

            # Add operation context and metrics
            op.add_context(
                query_embedding_dimension=len(query_embedding),
                raw_search_results=len(search_results),
                filtered_results=len(search_results),  # Already filtered by retriever
                avg_score=sum(doc.score for doc in search_results if doc.score) / len(search_results) if search_results else 0.0,
                filters_applied=bool(filters)
            )

            metrics.counter("queries_embedded_total", 1, component="retrieval_orchestrator")
            metrics.counter("vector_searches_total", 1, component="retrieval_orchestrator")
            metrics.counter("documents_found_total", len(search_results), component="retrieval_orchestrator", stage="raw")
            metrics.counter("documents_filtered_total", len(search_results), component="retrieval_orchestrator", stage="filtered")

            if search_results:
                metrics.histogram("retrieval_scores", [doc.score for doc in search_results if doc.score], component="retrieval_orchestrator")

            logger.info(
                "Retrieval orchestration completed successfully",
                query_length=len(query),
                embedding_dimension=len(query_embedding),
                raw_results=len(search_results),
                filtered_results=len(search_results),  # Already filtered by retriever
                score_threshold=score_threshold,
                component="retrieval_orchestrator"
            )

            return search_results

        except Exception as e:
            logger.error(
                "Retrieval orchestration failed",
                error=str(e),
                error_type=type(e).__name__,
                query_length=len(query),
                top_k=top_k,
                score_threshold=score_threshold,
                component="retrieval_orchestrator"
            )
            
            metrics.counter("retrieval_errors_total", 1, component="retrieval_orchestrator", error_type=type(e).__name__)
            
            raise RuntimeError(f"Retrieval failed: {str(e)}") from e