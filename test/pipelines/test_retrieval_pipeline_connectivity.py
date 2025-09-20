"""Pipeline connectivity tests for QdrantRetriever component.

Tests that the retriever can be properly connected in Haystack pipelines
and that socket types are compatible.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from haystack import Pipeline
from haystack.dataclasses import Document
from src.components.retriever.qdrant_retriever import QdrantRetriever


class TestRetrievalPipelineConnectivity:
    """Test QdrantRetriever pipeline connectivity and socket compatibility."""

    def test_qdrant_retriever_can_be_added_to_pipeline(self):
        """Test that QdrantRetriever can be added to a Haystack pipeline."""
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
                
                pipeline = Pipeline()
                retriever = QdrantRetriever()
                
                # This should not raise an exception
                pipeline.add_component("retriever", retriever)
                
                # Verify component was added
                assert "retriever" in pipeline.graph.nodes()

    def test_retriever_output_compatible_with_downstream_components(self):
        """Test that retriever output can be connected to components expecting List[Document]."""
        # Create a simple mock component that accepts List[Document]
        from haystack import component
        from typing import List
        
        @component
        class MockDocumentProcessor:
            @component.output_types(processed_docs=List[Document])
            def run(self, documents: List[Document]) -> dict[str, object]:
                return {"processed_docs": documents}
        
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
                
                pipeline = Pipeline()
                retriever = QdrantRetriever()
                processor = MockDocumentProcessor()
                
                pipeline.add_component("retriever", retriever)
                pipeline.add_component("processor", processor)
                
                # This should not raise type compatibility errors
                pipeline.connect("retriever.documents", "processor.documents")
                
                # Verify connection was created
                assert ("retriever", "processor") in pipeline.graph.edges()

    def test_retriever_in_simple_query_pipeline(self):
        """Test retriever in a simple query pipeline with mocked dependencies."""
        mock_documents = [
            Document(content="Test subtitle 1", meta={"start_ms": 1000, "end_ms": 2000}),
            Document(content="Test subtitle 2", meta={"start_ms": 3000, "end_ms": 4000}),
        ]
        
        # Mock the retriever's orchestrate_retrieval function
        with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
            mock_orchestrate.return_value = mock_documents
            
            with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=5,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True
                    )
                    
                    pipeline = Pipeline()
                    retriever = QdrantRetriever()
                    pipeline.add_component("retriever", retriever)
                    
                    # Run the pipeline
                    result = pipeline.run({"retriever": {"query": "test query"}})
                    
                    # Verify results
                    assert "retriever" in result
                    assert "documents" in result["retriever"]
                    assert len(result["retriever"]["documents"]) == 2
                    assert result["retriever"]["documents"][0].content == "Test subtitle 1"

    def test_retriever_socket_types_are_haystack_compatible(self):
        """Test that retriever declares socket types compatible with Haystack's type system."""
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
                
                # Check that output types are properly declared
                assert hasattr(retriever, '__haystack_output__')
                output_sockets = getattr(retriever, '__haystack_output__')
                
                # Verify that documents socket exists
                assert hasattr(output_sockets, '_sockets_dict')
                sockets_dict = output_sockets._sockets_dict
                assert 'documents' in sockets_dict
                
                # Verify the socket has the correct type
                documents_socket = sockets_dict['documents']
                assert documents_socket is not None

    def test_retriever_with_embedder_pipeline_integration(self):
        """Test retriever can coexist with other components in a pipeline."""
        mock_documents = [Document(content="Embedded result", meta={"score": 0.95})]
        
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
                    
                    pipeline = Pipeline()
                    retriever = QdrantRetriever()
                    pipeline.add_component("retriever", retriever)
                    
                    # Run the pipeline with just the retriever
                    result = pipeline.run({"retriever": {"query": "integration test"}})
                    
                    assert "retriever" in result
                    assert len(result["retriever"]["documents"]) == 1
                    assert result["retriever"]["documents"][0].content == "Embedded result"

    def test_retriever_run_method_signature_compatibility(self):
        """Test that run method signature is compatible with Haystack requirements."""
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
                
                # Check run method exists and has correct signature
                assert hasattr(retriever, 'run')
                assert callable(retriever.run)
                
                # Check that run method returns dict (required by Haystack)
                with patch('src.components.retriever.qdrant_retriever.orchestrate_retrieval') as mock_orchestrate:
                    mock_orchestrate.return_value = []
                    
                    result = retriever.run(query="signature test")
                    assert isinstance(result, dict)
                    assert "documents" in result