"""Tests for RAG pipeline components and functionality.

Tests the pipeline building, execution, and high-level RAGPipeline wrapper
with comprehensive mocking of dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.core.config import get_settings
from haystack.dataclasses import Document
from src.pipelines.rag_pipeline import (

    build_rag_pipeline,
    run_rag_query,
    RAGPipeline
)


class TestRAGPipelineBuilding:
    """Test RAG pipeline construction and configuration."""

    def test_build_rag_pipeline_component_initialization_default(self):
        """Test that components are initialized with default parameters."""
        with patch('src.pipelines.rag_pipeline.QdrantRetriever') as mock_retriever_class:
            with patch('src.pipelines.rag_pipeline.GroqGenerator') as mock_generator_class:
                with patch('src.pipelines.rag_pipeline.Pipeline') as mock_pipeline_class:
                    mock_pipeline = MagicMock()
                    mock_pipeline_class.return_value = mock_pipeline

                    result = build_rag_pipeline()

                    # Verify components were created with default parameters
                    mock_retriever_class.assert_called_once_with()
                    mock_generator_class.assert_called_once_with()

                    # Verify pipeline operations
                    mock_pipeline.add_component.assert_any_call("retriever", mock_retriever_class.return_value)
                    mock_pipeline.add_component.assert_any_call("generator", mock_generator_class.return_value)
                    mock_pipeline.connect.assert_called_once_with("retriever.documents", "generator.documents")

                    assert result == mock_pipeline

    def test_build_rag_pipeline_component_initialization_custom(self):
        """Test that components are initialized with custom parameters."""
        with patch('src.pipelines.rag_pipeline.QdrantRetriever') as mock_retriever_class:
            with patch('src.pipelines.rag_pipeline.GroqGenerator') as mock_generator_class:
                with patch('src.pipelines.rag_pipeline.Pipeline') as mock_pipeline_class:
                    mock_pipeline = MagicMock()
                    mock_pipeline_class.return_value = mock_pipeline

                    retriever_config = {"top_k": 15, "score_threshold": 0.8}
                    generator_config = {"model": "mixtral-8x7b-32768", "temperature": 0.3}

                    result = build_rag_pipeline(
                        retriever_config=retriever_config,
                        generator_config=generator_config
                    )

                    # Verify components were created with custom parameters
                    mock_retriever_class.assert_called_once_with(**retriever_config)
                    mock_generator_class.assert_called_once_with(**generator_config)

                    # Verify pipeline operations
                    mock_pipeline.add_component.assert_any_call("retriever", mock_retriever_class.return_value)
                    mock_pipeline.add_component.assert_any_call("generator", mock_generator_class.return_value)
                    mock_pipeline.connect.assert_called_once_with("retriever.documents", "generator.documents")

                    assert result == mock_pipeline


class TestRAGQueryExecution:
    """Test RAG query execution and parameter handling."""

    def create_mock_pipeline_result(self):
        """Helper to create mock pipeline results."""
        settings = get_settings()
        mock_documents = [
            Document(
                content="Technology should be used to help people.",
                meta={"speaker": "Tony Stark", "start_ms": 15000, "end_ms": 18000},
                score=0.95
            ),
            Document(
                content="Innovation drives progress in our society.",
                meta={"speaker": "Bruce Banner", "start_ms": 25000, "end_ms": 28000},
                score=0.87
            )
        ]

        mock_replies = ["Based on the context, Tony Stark believes technology should help people, while Bruce Banner emphasizes innovation's role in progress."]
        mock_meta = [{
            "model": settings.groq_model,
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 35,
                "total_tokens": 185
            }
        }]

        return {
            "retriever": {"documents": mock_documents},
            "generator": {"replies": mock_replies, "meta": mock_meta}
        }

    def test_run_rag_query_basic(self):
        """Test basic RAG query execution."""
        mock_pipeline = MagicMock()
        mock_result = self.create_mock_pipeline_result()
        
        with patch('src.pipelines.rag_pipeline.orchestrate_rag_query') as mock_orchestrate:
            mock_orchestrate.return_value = mock_result

            result = run_rag_query(mock_pipeline, "What do they think about technology?")

            # Verify orchestration was called
            mock_orchestrate.assert_called_once()
            call_args = mock_orchestrate.call_args[1]
            assert call_args["pipeline"] == mock_pipeline
            assert call_args["query"] == "What do they think about technology?"

            # Verify result structure
            assert result == mock_result

    def test_run_rag_query_with_parameters(self):
        """Test RAG query with parameter overrides."""
        mock_pipeline = MagicMock()
        mock_result = self.create_mock_pipeline_result()

        with patch('src.pipelines.rag_pipeline.orchestrate_rag_query') as mock_orchestrate:
            mock_orchestrate.return_value = mock_result

            result = run_rag_query(
                pipeline=mock_pipeline,
                query="Character analysis question",
                top_k=10,
                score_threshold=0.8,
                filters={"speaker": "Tony Stark"},
                task_type="character_analysis",
                custom_instructions="Focus on personality traits",
                max_tokens=512,
                temperature=0.3
            )

            # Verify parameters were passed through
            call_args = mock_orchestrate.call_args[1]
            assert call_args["top_k"] == 10
            assert call_args["score_threshold"] == 0.8
            assert call_args["filters"] == {"speaker": "Tony Stark"}
            assert call_args["task_type"] == "character_analysis"
            assert call_args["custom_instructions"] == "Focus on personality traits"
            assert call_args["max_tokens"] == 512
            assert call_args["temperature"] == 0.3

            assert result == mock_result


class TestRAGPipelineWrapper:
    """Test the high-level RAGPipeline wrapper class."""

    def test_rag_pipeline_initialization(self):
        """Test RAGPipeline initialization."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build:
            mock_pipeline = MagicMock()
            mock_build.return_value = mock_pipeline

            rag = RAGPipeline()

            # Verify pipeline was built
            mock_build.assert_called_once_with(
                retriever_config=None,
                generator_config=None
            )
            assert rag._pipeline == mock_pipeline

    def test_rag_pipeline_initialization_with_config(self):
        """Test RAGPipeline initialization with custom configurations."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build:
            mock_pipeline = MagicMock()
            mock_build.return_value = mock_pipeline

            retriever_config = {"top_k": 20}
            generator_config = {"model": "mixtral-8x7b-32768"}

            rag = RAGPipeline(
                retriever_config=retriever_config,
                generator_config=generator_config
            )

            # Verify configuration was passed through
            mock_build.assert_called_once_with(
                retriever_config=retriever_config,
                generator_config=generator_config
            )

    def test_rag_pipeline_query(self):
        """Test basic query through RAGPipeline wrapper."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build:
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_pipeline = MagicMock()
                mock_build.return_value = mock_pipeline
                
                mock_result = {"answer": "Test answer", "sources": []}
                mock_run.return_value = mock_result

                rag = RAGPipeline()
                result = rag.query("What is the meaning of life?")

                # Verify run_rag_query was called correctly
                mock_run.assert_called_once()
                call_args = mock_run.call_args
                assert call_args[1]["pipeline"] == mock_pipeline
                assert call_args[1]["query"] == "What is the meaning of life?"

                assert result == mock_result

    def test_rag_pipeline_character_analysis(self):
        """Test specialized character_analysis method."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline'):
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_result = {"answer": "Character analysis", "sources": []}
                mock_run.return_value = mock_result

                rag = RAGPipeline()
                result = rag.character_analysis(
                    "How does Tony Stark develop?",
                    character_name="Tony Stark"
                )

                # Verify correct parameters were passed
                call_args = mock_run.call_args[1]
                assert call_args["query"] == "How does Tony Stark develop?"
                assert call_args["task_type"] == "character_analysis"
                assert call_args["filters"] == {"speaker": "Tony Stark"}

                assert result == mock_result

    def test_rag_pipeline_find_quote(self):
        """Test specialized find_quote method."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline'):
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_result = {"answer": "Quote found", "sources": []}
                mock_run.return_value = mock_result

                rag = RAGPipeline()
                result = rag.find_quote(
                    "I am Iron Man",
                    speaker="Tony Stark"
                )

                # Verify correct parameters were passed
                call_args = mock_run.call_args[1]
                assert "Find this exact quote" in call_args["query"]
                assert "I am Iron Man" in call_args["query"]
                assert call_args["task_type"] == "quote_finding"
                assert call_args["filters"] == {"speaker": "Tony Stark"}
                assert call_args["score_threshold"] == 0.8  # Higher threshold for quotes

                assert result == mock_result

    def test_rag_pipeline_timeline_analysis(self):
        """Test specialized timeline_analysis method."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline'):
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_result = {"answer": "Timeline analysis", "sources": []}
                mock_run.return_value = mock_result

                rag = RAGPipeline()
                result = rag.timeline_analysis(
                    "What happens in the first act?",
                    time_range={"start_seconds": 0, "end_seconds": 1800}  # First 30 minutes
                )

                # Verify correct parameters were passed
                call_args = mock_run.call_args[1]
                assert call_args["query"] == "What happens in the first act?"
                assert call_args["task_type"] == "timeline"
                
                # Check time range filters
                filters = call_args["filters"]
                assert filters["start_ms"]["gte"] == 0  # 0 seconds * 1000
                assert filters["end_ms"]["lte"] == 1800000  # 1800 seconds * 1000

                assert result == mock_result

    def test_rag_pipeline_query_with_all_parameters(self):
        """Test query with all possible parameters."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline'):
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_result = {"answer": "Complete test", "sources": []}
                mock_run.return_value = mock_result

                rag = RAGPipeline()
                result = rag.query(
                    question="Complex question",
                    top_k=15,
                    score_threshold=0.7,
                    filters={"speaker": "Character A"},
                    task_type="scene_analysis",
                    custom_instructions="Be detailed",
                    max_tokens=1024,
                    temperature=0.2
                )

                # Verify all parameters were passed through
                call_args = mock_run.call_args[1]
                assert call_args["query"] == "Complex question"
                assert call_args["top_k"] == 15
                assert call_args["score_threshold"] == 0.7
                assert call_args["filters"] == {"speaker": "Character A"}
                assert call_args["task_type"] == "scene_analysis"
                assert call_args["custom_instructions"] == "Be detailed"
                assert call_args["max_tokens"] == 1024
                assert call_args["temperature"] == 0.2

                assert result == mock_result


class TestRAGPipelineErrorHandling:
    """Test error handling in RAG pipeline operations."""

    def test_run_rag_query_handles_orchestration_errors(self):
        """Test that run_rag_query properly handles orchestration errors."""
        mock_pipeline = MagicMock()

        with patch('src.pipelines.rag_pipeline.orchestrate_rag_query') as mock_orchestrate:
            mock_orchestrate.side_effect = RuntimeError("Orchestration failed")

            with pytest.raises(RuntimeError, match="Orchestration failed"):
                run_rag_query(mock_pipeline, "test query")

    def test_rag_pipeline_handles_build_errors(self):
        """Test that RAGPipeline handles pipeline building errors."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build:
            mock_build.side_effect = RuntimeError("Pipeline build failed")

            with pytest.raises(RuntimeError, match="Pipeline build failed"):
                RAGPipeline()

    def test_rag_pipeline_handles_query_errors(self):
        """Test that RAGPipeline handles query execution errors."""
        with patch('src.pipelines.rag_pipeline.build_rag_pipeline') as mock_build:
            with patch('src.pipelines.rag_pipeline.run_rag_query') as mock_run:
                mock_pipeline = MagicMock()
                mock_build.return_value = mock_pipeline
                mock_run.side_effect = RuntimeError("Query execution failed")

                rag = RAGPipeline()

                with pytest.raises(RuntimeError, match="Query execution failed"):
                    rag.query("test question")