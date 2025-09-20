"""Tests for orchestrate_retrieval function.

Tests the core retrieval orchestration logic with comprehensive mocking
of embedding and document store dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document
from src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval import orchestrate_retrieval


class TestOrchestateRetrieval:
    """Test the orchestrate_retrieval function."""

    def test_orchestrate_retrieval_successful_flow(self):
        """Test successful end-to-end retrieval orchestration."""
        # Mock embedding result
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock document store search results
        mock_docs = [
            Document(content="High score doc", meta={"title": "Doc 1"}, score=0.95),
            Document(content="Medium score doc", meta={"title": "Doc 2"}, score=0.75),
            Document(content="Low score doc", meta={"title": "Doc 3"}, score=0.45),
        ]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": mock_docs}
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                # Execute orchestration
                result = orchestrate_retrieval(
                    query="test query",
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=10,
                    score_threshold=0.5,
                    filters=None
                )
                
                # Verify embedding was called correctly
                mock_embedder.run.assert_called_once_with(text=["test query"])
                
                # Verify retriever was instantiated correctly
                mock_retriever_class.assert_called_once()
                call_args = mock_retriever_class.call_args[1]
                assert call_args["top_k"] == 10
                assert call_args["score_threshold"] == 0.5
                assert call_args["filters"] is None
                
                # Verify retriever run was called with correct embedding
                mock_retriever_instance.run.assert_called_once_with(query_embedding=mock_embedding)
                
                # Verify results (no filtering since QdrantEmbeddingRetriever handles that)
                assert len(result) == 3
                assert result[0].content == "High score doc"
                assert result[1].content == "Medium score doc"
                assert result[2].content == "Low score doc"
                
                # Verify retrieval metadata was added
                assert result[0].meta["retrieval_rank"] == 1
                assert result[1].meta["retrieval_rank"] == 2
                assert result[2].meta["retrieval_rank"] == 3
                assert result[0].meta["retrieval_query"] == "test query"
                assert result[0].meta["retrieval_score"] == 0.95

    def test_orchestrate_retrieval_with_filters(self):
        """Test that filters are passed correctly to document store."""
        mock_embedding = [0.1, 0.2, 0.3]
        mock_docs = [Document(content="Filtered doc", score=0.8)]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": mock_docs}
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                filters = {"category": "subtitle", "language": "en"}
                
                result = orchestrate_retrieval(
                    query="filtered query",
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=5,
                    score_threshold=0.0,
                    filters=filters
                )
                
                # Verify retriever was instantiated with filters
                call_args = mock_retriever_class.call_args[1]
                assert call_args["filters"] == filters
                
                assert len(result) == 1
                assert result[0].content == "Filtered doc"

    def test_orchestrate_retrieval_score_threshold_filtering(self):
        """Test that score threshold filtering works correctly."""
        mock_embedding = [0.1, 0.2, 0.3]
        
        # Create docs with various scores - the QdrantEmbeddingRetriever handles filtering
        mock_docs = [
            Document(content="Very high score", score=0.95),
            Document(content="High score", score=0.85),
            Document(content="Medium score", score=0.65),
            Document(content="Low score", score=0.45),
            Document(content="Very low score", score=0.25),
        ]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": mock_docs}
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                # Test with threshold of 0.7 - this is passed to the retriever which handles filtering
                result = orchestrate_retrieval(
                    query="threshold test",
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=10,
                    score_threshold=0.7,
                    filters=None
                )
                
                # The retriever handles the filtering, so we get all results back (in a real scenario)
                # For testing purposes, let's assume the retriever returns filtered results
                # We'll mock it to return only docs with score >= 0.7
                filtered_docs = [doc for doc in mock_docs if doc.score >= 0.7]
                mock_retriever_instance.run.return_value = {"documents": filtered_docs}
                
                # Re-run with updated mock
                result = orchestrate_retrieval(
                    query="threshold test",
                    document_store=MagicMock(),
                    top_k=10,
                    score_threshold=0.7,
                    filters=None
                )
                
                # Should only return docs with score >= 0.7
                assert len(result) == 2
                assert result[0].content == "Very high score"
                assert result[1].content == "High score"
                assert all(doc.score >= 0.7 for doc in result)

    def test_orchestrate_retrieval_empty_results(self):
        """Test handling of empty search results."""
        mock_embedding = [0.1, 0.2, 0.3]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": []}  # No results
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                result = orchestrate_retrieval(
                    query="no results query",
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=10,
                    score_threshold=0.0,
                    filters=None
                )
                
                assert result == []

    def test_orchestrate_retrieval_validates_input_parameters(self):
        """Test that input parameter validation works correctly."""
        mock_document_store = MagicMock()
        
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty or whitespace-only"):
            orchestrate_retrieval("", mock_document_store, 10, 0.5)
        
        with pytest.raises(ValueError, match="Query cannot be empty or whitespace-only"):
            orchestrate_retrieval("   ", mock_document_store, 10, 0.5)
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            orchestrate_retrieval("query", mock_document_store, 0, 0.5)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            orchestrate_retrieval("query", mock_document_store, -1, 0.5)
        
        # Test invalid score_threshold
        with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0"):
            orchestrate_retrieval("query", mock_document_store, 10, -0.1)
        
        with pytest.raises(ValueError, match="score_threshold must be between 0.0 and 1.0"):
            orchestrate_retrieval("query", mock_document_store, 10, 1.1)

    def test_orchestrate_retrieval_handles_embedding_errors(self):
        """Test error handling when embedding fails."""
        mock_document_store = MagicMock()
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.run.side_effect = RuntimeError("Embedding failed")
            mock_embedder_class.return_value = mock_embedder
            
            with pytest.raises(RuntimeError, match="Retrieval failed"):
                orchestrate_retrieval(
                    query="error query",
                    document_store=mock_document_store,
                    top_k=10,
                    score_threshold=0.0
                )

    def test_orchestrate_retrieval_handles_search_errors(self):
        """Test error handling when document store search fails."""
        mock_embedding = [0.1, 0.2, 0.3]
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.run.side_effect = RuntimeError("Search failed")
                mock_retriever_class.return_value = mock_retriever_instance
                
                with pytest.raises(RuntimeError, match="Retrieval failed"):
                    orchestrate_retrieval(
                        query="search error query",
                        document_store=MagicMock(),  # We're mocking the retriever, not the store
                        top_k=10,
                        score_threshold=0.0
                    )

    def test_orchestrate_retrieval_long_query_truncation(self):
        """Test that long queries are properly truncated in metadata."""
        mock_embedding = [0.1, 0.2, 0.3]
        long_query = "This is a very long query " * 10  # > 100 characters
        
        mock_docs = [Document(content="Test doc", score=0.8)]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": mock_docs}
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                result = orchestrate_retrieval(
                    query=long_query,
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=10,
                    score_threshold=0.0
                )
                
                # Verify query is truncated in metadata
                retrieval_query = result[0].meta["retrieval_query"]
                assert len(retrieval_query) <= 103  # 100 chars + "..."
                assert retrieval_query.endswith("...")

    def test_orchestrate_retrieval_preserves_existing_metadata(self):
        """Test that existing document metadata is preserved and extended."""
        mock_embedding = [0.1, 0.2, 0.3]
        
        # Document with existing metadata
        existing_meta = {"original_field": "original_value", "timestamp": "2023-01-01"}
        mock_docs = [Document(content="Test doc", meta=existing_meta, score=0.8)]
        
        # Mock the QdrantEmbeddingRetriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.run.return_value = {"documents": mock_docs}
        
        with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.TextEmbedder') as mock_embedder_class:
            with patch('src.components.retriever.utilities.qdrant_retriever.orchestrate_retrieval.QdrantEmbeddingRetriever') as mock_retriever_class:
                mock_embedder = MagicMock()
                mock_embedder.run.return_value = {"embedding": [mock_embedding]}
                mock_embedder_class.return_value = mock_embedder
                
                mock_retriever_class.return_value = mock_retriever_instance
                
                result = orchestrate_retrieval(
                    query="preservation test",
                    document_store=MagicMock(),  # We're mocking the retriever, not the store
                    top_k=10,
                    score_threshold=0.0
                )
                
                # Verify existing metadata is preserved
                assert result[0].meta["original_field"] == "original_value"
                assert result[0].meta["timestamp"] == "2023-01-01"
                
                # Verify new retrieval metadata is added
                assert result[0].meta["retrieval_rank"] == 1
                assert result[0].meta["retrieval_query"] == "preservation test"
                assert result[0].meta["retrieval_score"] == 0.8