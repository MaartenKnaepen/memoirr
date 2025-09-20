"""Tests for RAG query orchestration function.

Tests the core orchestration logic with comprehensive mocking
of pipeline execution and result processing.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack import Pipeline
from haystack.dataclasses import Document
from src.pipelines.utilities.rag_pipeline.orchestrate_rag_query import (
    orchestrate_rag_query,
    validate_rag_pipeline,
    _build_pipeline_inputs,
    _process_pipeline_result,
)


class TestOrchestrateRAGQuery:
    """Test the orchestrate_rag_query function."""

    def create_mock_pipeline_result(self):
        """Helper to create mock pipeline execution results."""
        mock_documents = [
            Document(
                content="Technology should be used responsibly.",
                meta={"speaker": "Tony Stark", "start_ms": 15000, "end_ms": 18000},
                score=0.95
            ),
            Document(
                content="Innovation is key to progress.",
                meta={"speaker": "Bruce Banner", "start_ms": 25000, "end_ms": 28000},
                score=0.87
            )
        ]

        mock_replies = ["Based on the context, the characters discuss responsible technology use and innovation."]
        mock_meta = [{
            "model": "llama3-8b-8192",
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 35,
                "total_tokens": 185
            },
            "query": "What do they think about technology?",
            "document_count": 2
        }]

        return {
            "retriever": {"documents": mock_documents},
            "generator": {"replies": mock_replies, "meta": mock_meta}
        }

    def test_orchestrate_rag_query_successful_flow(self):
        """Test successful end-to-end RAG query orchestration."""
        mock_pipeline = MagicMock()
        mock_pipeline_result = self.create_mock_pipeline_result()
        mock_pipeline.run.return_value = mock_pipeline_result

        # Create a mock pipeline graph
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]

        query = "What do the characters think about technology?"

        result = orchestrate_rag_query(
            pipeline=mock_pipeline,
            query=query,
            top_k=5,
            task_type="character_analysis"
        )

        # Verify pipeline was called with correct inputs
        mock_pipeline.run.assert_called_once()
        call_args = mock_pipeline.run.call_args[0][0]

        # Check retriever inputs
        assert call_args["retriever"]["query"] == query
        assert call_args["retriever"]["top_k"] == 5

        # Check generator inputs
        assert call_args["generator"]["query"] == query
        assert call_args["generator"]["task_type"] == "character_analysis"

        # Verify result structure
        assert "retriever" in result
        assert "generator" in result
        assert "summary" in result
        assert "answer" in result
        assert "sources" in result

        # Check summary information
        summary = result["summary"]
        assert summary["query"] == query
        assert summary["documents_retrieved"] == 2
        assert summary["replies_generated"] == 1
        assert summary["has_results"] is True

    def test_orchestrate_rag_query_with_all_parameters(self):
        """Test orchestration with all possible parameter overrides."""
        mock_pipeline = MagicMock()
        mock_pipeline_result = self.create_mock_pipeline_result()
        mock_pipeline.run.return_value = mock_pipeline_result
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]

        result = orchestrate_rag_query(
            pipeline=mock_pipeline,
            query="Complex question",
            top_k=10,
            score_threshold=0.8,
            filters={"speaker": "Tony Stark"},
            task_type="quote_finding",
            custom_instructions="Be precise",
            max_tokens=512,
            temperature=0.3
        )

        # Verify all parameters were passed to pipeline
        call_args = mock_pipeline.run.call_args[0][0]

        # Retriever parameters
        retriever_inputs = call_args["retriever"]
        assert retriever_inputs["query"] == "Complex question"
        assert retriever_inputs["top_k"] == 10
        assert retriever_inputs["filters"] == {"speaker": "Tony Stark"}

        # Generator parameters
        generator_inputs = call_args["generator"]
        assert generator_inputs["query"] == "Complex question"
        assert generator_inputs["task_type"] == "quote_finding"
        assert generator_inputs["custom_instructions"] == "Be precise"
        assert generator_inputs["max_tokens"] == 512
        assert generator_inputs["temperature"] == 0.3

        # Should still return processed results
        assert "answer" in result
        assert "sources" in result

    def test_orchestrate_rag_query_validates_input(self):
        """Test that input parameter validation works correctly."""
        mock_pipeline = MagicMock()

        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrate_rag_query(mock_pipeline, "")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrate_rag_query(mock_pipeline, "   ")

    def test_orchestrate_rag_query_handles_pipeline_errors(self):
        """Test error handling when pipeline execution fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("Pipeline execution failed")
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]

        with pytest.raises(RuntimeError, match="RAG query failed"):
            orchestrate_rag_query(mock_pipeline, "test query")

    def test_orchestrate_rag_query_handles_processing_errors(self):
        """Test error handling when result processing fails."""
        mock_pipeline = MagicMock()
        
        # Return invalid pipeline result (missing required keys)
        mock_pipeline.run.return_value = {"invalid": "result"}
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]

        with pytest.raises(RuntimeError, match="RAG query failed"):
            orchestrate_rag_query(mock_pipeline, "test query")


class TestBuildPipelineInputs:
    """Test the _build_pipeline_inputs helper function."""

    def test_build_pipeline_inputs_basic(self):
        """Test building pipeline inputs with basic query."""
        inputs = _build_pipeline_inputs("What is the answer?")

        assert "retriever" in inputs
        assert "generator" in inputs

        # Both should have the query
        assert inputs["retriever"]["query"] == "What is the answer?"
        assert inputs["generator"]["query"] == "What is the answer?"

        # Should only have query for basic case
        assert len(inputs["retriever"]) == 1
        assert len(inputs["generator"]) == 1

    def test_build_pipeline_inputs_with_retrieval_params(self):
        """Test building inputs with retrieval-specific parameters."""
        inputs = _build_pipeline_inputs(
            query="Test query",
            top_k=15,
            score_threshold=0.7,
            filters={"speaker": "Character A"}
        )

        retriever_inputs = inputs["retriever"]
        assert retriever_inputs["query"] == "Test query"
        assert retriever_inputs["top_k"] == 15
        assert retriever_inputs["filters"] == {"speaker": "Character A"}

        # Generator should only have query
        generator_inputs = inputs["generator"]
        assert generator_inputs["query"] == "Test query"
        assert len(generator_inputs) == 1

    def test_build_pipeline_inputs_with_generation_params(self):
        """Test building inputs with generation-specific parameters."""
        inputs = _build_pipeline_inputs(
            query="Test query",
            task_type="character_analysis",
            custom_instructions="Be detailed",
            max_tokens=1024,
            temperature=0.2
        )

        # Retriever should only have query
        retriever_inputs = inputs["retriever"]
        assert retriever_inputs["query"] == "Test query"
        assert len(retriever_inputs) == 1

        # Generator should have all parameters
        generator_inputs = inputs["generator"]
        assert generator_inputs["query"] == "Test query"
        assert generator_inputs["task_type"] == "character_analysis"
        assert generator_inputs["custom_instructions"] == "Be detailed"
        assert generator_inputs["max_tokens"] == 1024
        assert generator_inputs["temperature"] == 0.2

    def test_build_pipeline_inputs_none_values_excluded(self):
        """Test that None values are excluded from pipeline inputs."""
        inputs = _build_pipeline_inputs(
            query="Test query",
            top_k=None,
            task_type=None,
            max_tokens=512  # This should be included
        )

        # Should not include None values
        assert "top_k" not in inputs["retriever"]
        assert "task_type" not in inputs["generator"]

        # Should include non-None values
        assert inputs["generator"]["max_tokens"] == 512


class TestProcessPipelineResult:
    """Test the _process_pipeline_result helper function."""

    def create_valid_pipeline_result(self):
        """Helper to create valid pipeline result for testing."""
        documents = [
            Document(
                content="Test content for processing",
                meta={"speaker": "Test Speaker", "start_ms": 1000},
                score=0.9
            )
        ]

        return {
            "retriever": {"documents": documents},
            "generator": {
                "replies": ["Test reply"],
                "meta": [{"model": "test-model", "usage": {"total_tokens": 100}}]
            }
        }

    def test_process_pipeline_result_valid_input(self):
        """Test processing valid pipeline results."""
        pipeline_result = self.create_valid_pipeline_result()
        original_query = "What is the test about?"

        processed = _process_pipeline_result(pipeline_result, original_query)

        # Verify structure
        assert "retriever" in processed
        assert "generator" in processed
        assert "summary" in processed
        assert "answer" in processed
        assert "sources" in processed

        # Check summary
        summary = processed["summary"]
        assert summary["query"] == original_query
        assert summary["documents_retrieved"] == 1
        assert summary["replies_generated"] == 1
        assert summary["has_results"] is True
        assert summary["best_document_score"] == 0.9

        # Check answer extraction
        assert processed["answer"] == "Test reply"

        # Check sources
        sources = processed["sources"]
        assert len(sources) == 1
        assert sources[0]["score"] == 0.9
        assert "Test content" in sources[0]["content"]

    def test_process_pipeline_result_multiple_replies(self):
        """Test processing results with multiple replies."""
        pipeline_result = self.create_valid_pipeline_result()
        # Add multiple replies
        pipeline_result["generator"]["replies"] = ["Primary answer", "Alternative answer"]

        processed = _process_pipeline_result(pipeline_result, "test query")

        # Primary answer should be extracted
        assert processed["answer"] == "Primary answer"
        
        # Alternative answers should be available
        assert "alternative_answers" in processed
        assert processed["alternative_answers"] == ["Alternative answer"]

    def test_process_pipeline_result_missing_retriever(self):
        """Test error handling when retriever results are missing."""
        pipeline_result = {
            "generator": {"replies": ["Test"], "meta": [{}]}
            # Missing "retriever"
        }

        with pytest.raises(RuntimeError, match="Missing retriever results"):
            _process_pipeline_result(pipeline_result, "test query")

    def test_process_pipeline_result_missing_generator(self):
        """Test error handling when generator results are missing."""
        pipeline_result = {
            "retriever": {"documents": []}
            # Missing "generator"
        }

        with pytest.raises(RuntimeError, match="Missing generator results"):
            _process_pipeline_result(pipeline_result, "test query")

    def test_process_pipeline_result_invalid_documents(self):
        """Test error handling when documents format is invalid."""
        pipeline_result = {
            "retriever": {"documents": "not a list"},  # Should be list
            "generator": {"replies": ["Test"], "meta": [{}]}
        }

        with pytest.raises(RuntimeError, match="Documents should be a list"):
            _process_pipeline_result(pipeline_result, "test query")

    def test_process_pipeline_result_invalid_replies(self):
        """Test error handling when replies format is invalid."""
        pipeline_result = {
            "retriever": {"documents": []},
            "generator": {"replies": "not a list", "meta": [{}]}  # Should be list
        }

        with pytest.raises(RuntimeError, match="Replies should be a list"):
            _process_pipeline_result(pipeline_result, "test query")

    def test_process_pipeline_result_empty_results(self):
        """Test processing empty but valid results."""
        pipeline_result = {
            "retriever": {"documents": []},
            "generator": {"replies": [], "meta": []}
        }

        processed = _process_pipeline_result(pipeline_result, "test query")

        # Should handle empty results gracefully
        summary = processed["summary"]
        assert summary["documents_retrieved"] == 0
        assert summary["replies_generated"] == 0
        assert summary["has_results"] is False
        assert summary["best_document_score"] == 0.0

        # No answer should be extracted
        assert "answer" not in processed
        assert "sources" not in processed or processed["sources"] == []


class TestValidateRAGPipeline:
    """Test the validate_rag_pipeline function."""

    def test_validate_rag_pipeline_valid(self):
        """Test validation of properly configured pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator", "other_component"]
        mock_pipeline.graph.edges.return_value = [("retriever", "generator"), ("other", "connections")]

        result = validate_rag_pipeline(mock_pipeline)

        assert result is True

    def test_validate_rag_pipeline_missing_retriever(self):
        """Test validation failure when retriever is missing."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph.nodes.return_value = ["generator", "other_component"]

        with pytest.raises(ValueError, match="Pipeline missing 'retriever' component"):
            validate_rag_pipeline(mock_pipeline)

    def test_validate_rag_pipeline_missing_generator(self):
        """Test validation failure when generator is missing."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph.nodes.return_value = ["retriever", "other_component"]

        with pytest.raises(ValueError, match="Pipeline missing 'generator' component"):
            validate_rag_pipeline(mock_pipeline)

    def test_validate_rag_pipeline_not_connected(self):
        """Test validation failure when components are not connected."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]
        mock_pipeline.graph.edges.return_value = [("other", "connections")]  # No retriever->generator

        with pytest.raises(ValueError, match="Pipeline components not properly connected"):
            validate_rag_pipeline(mock_pipeline)

    def test_validate_rag_pipeline_connection_variations(self):
        """Test that validation accepts different connection patterns."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]
        
        # Test direct connection
        mock_pipeline.graph.edges.return_value = [("retriever", "generator")]
        assert validate_rag_pipeline(mock_pipeline) is True

        # Test connection among other edges
        mock_pipeline.graph.edges.return_value = [
            ("other", "component"),
            ("retriever", "generator"),
            ("more", "edges")
        ]
        assert validate_rag_pipeline(mock_pipeline) is True