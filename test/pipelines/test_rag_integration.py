"""Integration tests for RAG pipeline.

Tests the complete end-to-end flow from query to answer with real component
interactions (mocked external dependencies like Qdrant and Groq APIs).
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack import Pipeline
from haystack.dataclasses import Document
from src.pipelines.rag_pipeline import build_rag_pipeline, RAGPipeline
from src.core.config import get_settings


class TestRAGPipelineIntegration:
    """Integration tests for complete RAG pipeline functionality."""

    def create_mock_subtitle_documents(self):
        """Create realistic subtitle documents for testing."""
        return [
            Document(
                content="I think technology should be used to help people, not harm them.",
                meta={
                    "start_ms": 15000,
                    "end_ms": 19000,
                    "caption_index": 5,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.95
                },
                score=0.95
            ),
            Document(
                content="Innovation is the key to solving our biggest challenges.",
                meta={
                    "start_ms": 45000,
                    "end_ms": 48000,
                    "caption_index": 15,
                    "speaker": "Bruce Banner",
                    "retrieval_score": 0.87
                },
                score=0.87
            ),
            Document(
                content="We need to be responsible with the power we have.",
                meta={
                    "start_ms": 67000,
                    "end_ms": 70000,
                    "caption_index": 22,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.82
                },
                score=0.82
            )
        ]

    def create_mock_groq_response(self, content="Based on the context provided, Tony Stark and Bruce Banner discuss the responsible use of technology and innovation to solve challenges."):
        """Create mock Groq API response."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content=content),
                finish_reason="stop",
                index=0
            )
        ]
        response.usage = MagicMock(
            prompt_tokens=180,
            completion_tokens=45,
            total_tokens=225
        )
        response.id = "integration_test_response_123"
        response.created = 1234567890
        response.object = "chat.completion"
        return response

    def test_end_to_end_rag_query_character_analysis(self):
        """Test complete RAG flow for character analysis query."""
        # Get actual settings to use in mock data
        settings = get_settings()
        
        mock_documents = self.create_mock_subtitle_documents()
        
        # Mock the entire RAG pipeline execution
        with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run_rag:
            mock_result = {
                "retriever": {"documents": mock_documents},
                "generator": {
                    "replies": ["Tony Stark demonstrates a strong ethical stance on technology."],
                    "meta": [{"model": settings.groq_model, "usage": {"total_tokens": 100}}]
                },
                "summary": {
                    "query": "How do Tony Stark and Bruce Banner view technology and responsibility?",
                    "documents_retrieved": 3,
                    "replies_generated": 1,
                    "has_results": True
                },
                "answer": "Tony Stark demonstrates a strong ethical stance on technology.",
                "sources": [
                    {"content": "I think technology should be used...", "score": 0.95},
                    {"content": "Innovation is the key to solving...", "score": 0.87},
                    {"content": "We need to be responsible...", "score": 0.82}
                ]
            }
            mock_run_rag.return_value = mock_result

            # Create and test RAG pipeline
            rag = RAGPipeline()
            
            result = rag.character_analysis(
                "How do Tony Stark and Bruce Banner view technology and responsibility?",
                character_name="Tony Stark"
            )

            # Verify run_rag_query was called with correct parameters
            mock_run_rag.assert_called_once()
            call_args = mock_run_rag.call_args[1]
            assert call_args["query"] == "How do Tony Stark and Bruce Banner view technology and responsibility?"
            assert call_args["task_type"] == "character_analysis"
            assert call_args["filters"] == {"speaker": "Tony Stark"}

            # Verify result structure
            assert "answer" in result
            assert "sources" in result
            assert result["summary"]["documents_retrieved"] == 3
            assert "Tony Stark" in result["answer"]

    def test_end_to_end_rag_query_quote_finding(self):
        """Test complete RAG flow for quote finding."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        # Create documents with specific quote
        quote_documents = [
            Document(
                content="I am Iron Man.",
                meta={
                    "start_ms": 120000,
                    "end_ms": 121500,
                    "caption_index": 45,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.98
                },
                score=0.98
            ),
            Document(
                content="I am not Tony Stark, I am Iron Man.",
                meta={
                    "start_ms": 118000,
                    "end_ms": 120000,
                    "caption_index": 44,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.92
                },
                score=0.92
            )
        ]

        mock_groq_response = self.create_mock_groq_response(
            "The exact quote is 'I am Iron Man' spoken by Tony Stark at timestamp 2:00-2:01.5. "
            "This is the iconic moment where Tony Stark reveals his secret identity."
        )

        # Create a mock pipeline with our mocked components
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["retriever", "generator"]
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.graph = mock_graph
        mock_pipeline.run.return_value = {
            "retriever": {"documents": quote_documents},
            "generator": {
                "replies": [mock_groq_response.choices[0].message.content],
                "meta": [{
                    "model": settings.groq_model,
                    "usage": {
                        "prompt_tokens": mock_groq_response.usage.prompt_tokens,
                        "completion_tokens": mock_groq_response.usage.completion_tokens,
                        "total_tokens": mock_groq_response.usage.total_tokens
                    }
                }]
            }
        }

        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build_pipeline:
            mock_build_pipeline.return_value = mock_pipeline
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True,
                        groq_model=settings.groq_model,
                        groq_system_prompt_template="default_system.j2",
                        groq_max_tokens=1024,
                        groq_temperature=0.7,
                        groq_top_p=1.0,
                        groq_stream=False,
                        groq_api_key="test_key",
                        groq_max_context_length=4000
                    )

                    mock_groq_client = MagicMock()
                    mock_groq_client.chat.completions.create.return_value = mock_groq_response
                    mock_groq_class.return_value = mock_groq_client

                    rag = RAGPipeline()
                    
                    result = rag.find_quote(
                        "I am Iron Man",
                        speaker="Tony Stark"
                    )

                    # Verify pipeline was called with correct inputs
                    mock_pipeline.run.assert_called_once()
                    call_args = mock_pipeline.run.call_args[0][0]
                    assert "Find this exact quote or similar dialogue: I am Iron Man" in call_args["retriever"]["query"]
                    assert call_args["retriever"]["filters"]["speaker"] == "Tony Stark"
                    # top_k is not passed explicitly, so it shouldn't be in the inputs
                    assert "top_k" not in call_args["retriever"]
                    assert call_args["generator"]["task_type"] == "quote_finding"

                    # Verify result contains the quote
                    assert "I am Iron Man" in result["answer"]

    def test_end_to_end_rag_query_timeline_analysis(self):
        """Test complete RAG flow for timeline analysis."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        timeline_documents = [
            Document(
                content="The journey begins as we see our hero preparing for the mission.",
                meta={
                    "start_ms": 300000,  # 5 minutes
                    "end_ms": 305000,
                    "caption_index": 10,
                    "speaker": "Narrator",
                    "retrieval_score": 0.89
                },
                score=0.89
            ),
            Document(
                content="Twenty minutes later, the first challenge appears.",
                meta={
                    "start_ms": 1200000,  # 20 minutes
                    "end_ms": 1205000,
                    "caption_index": 35,
                    "speaker": "Narrator",
                    "retrieval_score": 0.85
                },
                score=0.85
            )
        ]

        mock_groq_response = self.create_mock_groq_response(
            "Based on the timeline, the story progression is as follows: "
            "At 5 minutes, the hero prepares for the mission. "
            "At 20 minutes, the first major challenge is encountered. "
            "This shows a deliberate pacing in the narrative structure."
        )

        # Create a mock pipeline with our mocked components
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["retriever", "generator"]
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.graph = mock_graph
        mock_pipeline.run.return_value = {
            "retriever": {"documents": timeline_documents},
            "generator": {
                "replies": [mock_groq_response.choices[0].message.content],
                "meta": [{
                    "model": settings.groq_model,
                    "usage": {
                        "prompt_tokens": mock_groq_response.usage.prompt_tokens,
                        "completion_tokens": mock_groq_response.usage.completion_tokens,
                        "total_tokens": mock_groq_response.usage.total_tokens
                    }
                }]
            }
        }

        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build_pipeline:
            mock_build_pipeline.return_value = mock_pipeline
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                with patch('src.core.config.get_settings') as mock_settings:
                    mock_settings.return_value = MagicMock(
                        retrieval_top_k=10,
                        retrieval_score_threshold=0.0,
                        retrieval_return_embedding=False,
                        qdrant_url="http://localhost:6300",
                        qdrant_collection="test_collection",
                        qdrant_wait_result=True,
                        groq_model=settings.groq_model,
                        groq_system_prompt_template="default_system.j2",
                        groq_max_tokens=1024,
                        groq_temperature=0.7,
                        groq_top_p=1.0,
                        groq_stream=False,
                        groq_api_key="test_key",
                        groq_max_context_length=4000
                    )

                    mock_groq_client = MagicMock()
                    mock_groq_client.chat.completions.create.return_value = mock_groq_response
                    mock_groq_class.return_value = mock_groq_client

                    rag = RAGPipeline()
                    
                    result = rag.timeline_analysis(
                        "What happens in the first 30 minutes?",
                        time_range={"start_seconds": 0, "end_seconds": 1800}  # 0-30 minutes
                    )

                    # Verify pipeline was called with correct inputs
                    mock_pipeline.run.assert_called_once()
                    call_args = mock_pipeline.run.call_args[0][0]
                    assert "What happens in the first 30 minutes?" in call_args["retriever"]["query"]
                    assert call_args["retriever"]["filters"]["start_ms"]["gte"] == 0
                    assert call_args["retriever"]["filters"]["end_ms"]["lte"] == 1800000  # 30 minutes in ms
                    assert call_args["generator"]["task_type"] == "timeline"

                    # Verify result discusses timing
                    assert "minutes" in result["answer"]

    def test_end_to_end_error_handling_no_documents_found(self):
        """Test pipeline behavior when no documents are retrieved."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        mock_groq_response = self.create_mock_groq_response(
            "I don't have enough context to answer that question based on the provided subtitles."
        )

        # Create a mock pipeline with our mocked components
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["retriever", "generator"]
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.graph = mock_graph
        mock_pipeline.run.return_value = {
            "retriever": {"documents": []},  # No documents found
            "generator": {
                "replies": [mock_groq_response.choices[0].message.content],
                "meta": [{
                    "model": settings.groq_model,
                    "usage": {
                        "prompt_tokens": mock_groq_response.usage.prompt_tokens,
                        "completion_tokens": mock_groq_response.usage.completion_tokens,
                        "total_tokens": mock_groq_response.usage.total_tokens
                    }
                }]
            }
        }

        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build_pipeline:
            mock_build_pipeline.return_value = mock_pipeline

            rag = RAGPipeline()
            
            result = rag.query("Question with no relevant content")

            # Verify pipeline was called
            mock_pipeline.run.assert_called_once()

            # Pipeline should still complete successfully
            assert "answer" in result
            assert result["summary"]["documents_retrieved"] == 0
            assert result["summary"]["has_results"] is True  # Still has generated answer
            # sources key should not be present when no documents are found
            assert "sources" not in result

    def test_pipeline_component_validation(self):
        """Test that pipeline validation works correctly."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        from src.pipelines.utilities.rag_pipeline.orchestrate_rag_query import validate_rag_pipeline

        # Build a real pipeline and validate it
        with patch('src.components.retriever.qdrant_retriever.QdrantDocumentStore'):
            with patch('src.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock(
                    retrieval_top_k=10,
                    retrieval_score_threshold=0.0,
                    retrieval_return_embedding=False,
                    qdrant_url="http://localhost:6300",
                    qdrant_collection="test_collection",
                    qdrant_wait_result=True,
                    groq_model=settings.groq_model,
                    groq_system_prompt_template="default_system.j2",
                    groq_max_tokens=1024,
                    groq_temperature=0.7,
                    groq_top_p=1.0,
                    groq_stream=False,
                    groq_api_key="test_key",
                    groq_max_context_length=4000
                )

                pipeline = build_rag_pipeline()
                
                # Validation should pass
                assert validate_rag_pipeline(pipeline) is True

                # Check pipeline structure
                nodes = list(pipeline.graph.nodes())
                assert "retriever" in nodes
                assert "generator" in nodes

                edges = list(pipeline.graph.edges())
                assert ("retriever", "generator") in edges