"""Tests for QdrantRetriever Haystack component.

Tests the component interface, configuration, and integration with mocked dependencies.
Follows Memoirr testing patterns: comprehensive mocking, error scenarios, metrics validation.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document
from src.components.retriever.qdrant_retriever import QdrantRetriever


class TestQdrantRetrieverComponent:
    """Test the QdrantRetriever Haystack component."""

    def test_qdrant_retriever_initializes_with_defaults(self):
        """Test that component initializes with default configuration."""
        with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            with patch('src.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock(
                    retrieval_top_k=10,
                    retrieval_score_threshold=0.0,
                    retrieval_return_embedding=False,
                    qdrant_url="http://localhost:6300",
                    qdrant_collection="test_collection",
                    qdrant_wait_result=True
                )
                
                retriever = QdrantRetriever()
                
                assert retriever.top_k == 10
                assert retriever.score_threshold == 0.0
                assert retriever.return_embedding is False
                assert retriever.filters == {}

    def test_qdrant_retriever_initializes_with_custom_params(self):
        """Test that component respects custom initialization parameters."""
        with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            with patch('src.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock(
                    qdrant_url="http://localhost:6300",
                    qdrant_collection="test_collection",
                    qdrant_wait_result=True
                )
                
                custom_filters = {"type": "subtitle"}
                retriever = QdrantRetriever(
                    top_k=20,
                    score_threshold=0.5,
                    return_embedding=True,
                    filters=custom_filters
                )
                
                assert retriever.top_k == 20
                assert retriever.score_threshold == 0.5
                assert retriever.return_embedding is True
                assert retriever.filters == custom_filters

    def test_qdrant_retriever_run_returns_documents(self):
        """Test that run method retrieves and returns documents successfully."""
        # Mock the orchestrate_retrieval function
        mock_documents = [
            Document(content="Test document 1", meta={"score": 0.95}),
            Document(content="Test document 2", meta={"score": 0.87}),
        ]
        
        with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
            mock_orchestrate.return_value = mock_documents
            
            with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True
                    )
                    
                    retriever = QdrantRetriever()
                    result = retriever.run(query="test query")
                    
                    assert "documents" in result
                    assert len(result["documents"]) == 2
                    assert result["documents"] == mock_documents
                    
                    # Verify orchestrate_retrieval was called with correct parameters
                    mock_orchestrate.assert_called_once()
                    call_args = mock_orchestrate.call_args[1]
                    assert call_args["query"] == "test query"
                    assert call_args["top_k"] == 10
                    assert call_args["score_threshold"] == 0.0
                    assert call_args["filters"] == {}

    def test_qdrant_retriever_run_with_overrides(self):
        """Test that run method respects parameter overrides."""
        mock_documents = [Document(content="Test document", meta={"score": 0.95})]
        
        with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
            mock_orchestrate.return_value = mock_documents
            
            with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True
                    )
                    
                    retriever = QdrantRetriever()
                    custom_filters = {"category": "important"}
                    result = retriever.run(
                        query="override test",
                        top_k=5,
                        filters=custom_filters
                    )
                    
                    assert "documents" in result
                    assert result["documents"] == mock_documents
                    
                    # Verify overrides were passed to orchestrator
                    call_args = mock_orchestrate.call_args[1]
                    assert call_args["query"] == "override test"
                    assert call_args["top_k"] == 5
                    assert call_args["filters"] == custom_filters

    def test_qdrant_retriever_run_with_empty_results(self):
        """Test that run method handles empty search results gracefully."""
        with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
            mock_orchestrate.return_value = []  # No documents found
            
            with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True
                    )
                    
                    retriever = QdrantRetriever()
                    result = retriever.run(query="no results query")
                    
                    assert "documents" in result
                    assert result["documents"] == []
                    assert len(result["documents"]) == 0

    def test_qdrant_retriever_run_handles_orchestrator_errors(self):
        """Test that run method properly handles and propagates orchestrator errors."""
        with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
            mock_orchestrate.side_effect = RuntimeError("Retrieval failed")
            
            with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True
                    )
                    
                    retriever = QdrantRetriever()
                    
                    with pytest.raises(RuntimeError, match="Retrieval failed"):
                        retriever.run(query="error query")

    def test_qdrant_retriever_component_output_types(self):
        """Test that component declares correct output types for Haystack compatibility."""
        with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
            with patch('src.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock(
                    retrieval_top_k=10,
                    retrieval_score_threshold=0.0,
                    retrieval_return_embedding=False,
                    qdrant_url="http://localhost:6300",
                    qdrant_collection="test_collection",
                    qdrant_wait_result=True
                )
                
                retriever = QdrantRetriever()
                
                # Check that the component has the expected output types
                # This validates Haystack component interface compliance
                assert hasattr(retriever, '__haystack_output__')
                output_types = getattr(retriever, '__haystack_output__')
                assert 'documents' in output_types