"""Tests for baseline RAG + evaluation pipeline integration.

Tests the combined RAG and evaluation pipeline following Memoirr standards
with comprehensive mocking and error handling validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from haystack import Pipeline
from haystack.components.evaluators import FaithfulnessEvaluator, ContextRelevanceEvaluator

from src.evaluation.pipelines.baseline_pipeline import (
    build_rag_with_evaluation_pipeline,
    run_baseline_evaluation_pipeline,
    _calculate_summary_metrics
)


class TestBuildRagWithEvaluationPipeline:
    """Test cases for RAG + evaluation pipeline construction."""

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    @patch('src.evaluation.pipelines.baseline_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.baseline_pipeline.ContextRelevanceEvaluator')
    def test_build_rag_with_evaluation_pipeline_default_config(
        self,
        mock_context_evaluator,
        mock_faithfulness_evaluator,
        mock_build_rag
    ):
        """Test RAG + evaluation pipeline construction with default configuration."""
        # ARRANGE
        mock_rag_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["retriever", "generator", "faithfulness_eval", "context_eval"]
        mock_rag_pipeline.graph = mock_graph
        mock_build_rag.return_value = mock_rag_pipeline
        
        mock_faithfulness_instance = Mock()
        mock_context_instance = Mock()
        mock_faithfulness_evaluator.return_value = mock_faithfulness_instance
        mock_context_evaluator.return_value = mock_context_instance
        
        # ACT
        result = build_rag_with_evaluation_pipeline()
        
        # ASSERT
        assert result == mock_rag_pipeline
        
        # Verify RAG pipeline is built with default configs
        mock_build_rag.assert_called_once_with(
            retriever_config=None,
            generator_config=None
        )
        
        # Verify evaluators are created and added
        mock_faithfulness_evaluator.assert_called_once()
        mock_context_evaluator.assert_called_once()
        
        mock_rag_pipeline.add_component.assert_any_call(
            "faithfulness_eval", mock_faithfulness_instance
        )
        mock_rag_pipeline.add_component.assert_any_call(
            "context_eval", mock_context_instance
        )

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    def test_build_rag_with_evaluation_pipeline_custom_configs(self, mock_build_rag):
        """Test RAG + evaluation pipeline construction with custom configurations."""
        # ARRANGE
        mock_rag_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["retriever", "generator", "faithfulness_eval", "context_eval"]
        mock_rag_pipeline.graph = mock_graph
        mock_build_rag.return_value = mock_rag_pipeline
        
        retriever_config = {"top_k": 10, "score_threshold": 0.7}
        generator_config = {"max_tokens": 500, "temperature": 0.2}
        
        # ACT
        result = build_rag_with_evaluation_pipeline(
            retriever_config=retriever_config,
            generator_config=generator_config
        )
        
        # ASSERT
        mock_build_rag.assert_called_once_with(
            retriever_config=retriever_config,
            generator_config=generator_config
        )

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    def test_build_rag_with_evaluation_pipeline_evaluation_disabled(self, mock_build_rag):
        """Test pipeline construction with evaluation disabled."""
        # ARRANGE
        mock_rag_pipeline = Mock(spec=Pipeline)
        mock_build_rag.return_value = mock_rag_pipeline
        
        # ACT
        result = build_rag_with_evaluation_pipeline(enable_evaluation=False)
        
        # ASSERT
        assert result == mock_rag_pipeline
        # Verify no evaluation components are added
        mock_rag_pipeline.add_component.assert_not_called()

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    @patch('src.evaluation.pipelines.baseline_pipeline.get_logger')
    def test_build_rag_with_evaluation_pipeline_logs_success(self, mock_get_logger, mock_build_rag):
        """Test that pipeline construction logs success message."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_rag_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["retriever", "generator", "faithfulness_eval"]
        mock_rag_pipeline.graph = mock_graph
        mock_build_rag.return_value = mock_rag_pipeline
        
        # ACT
        result = build_rag_with_evaluation_pipeline()
        
        # ASSERT
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["component"] == "baseline_pipeline"
        assert "total_components" in log_call_args
        assert "evaluation_enabled" in log_call_args

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    @patch('src.evaluation.pipelines.baseline_pipeline.get_logger')
    def test_build_rag_with_evaluation_pipeline_handles_exceptions(
        self, 
        mock_get_logger, 
        mock_build_rag
    ):
        """Test that pipeline construction handles and logs exceptions."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_build_rag.side_effect = RuntimeError("RAG pipeline creation failed")
        
        # ACT & ASSERT
        with pytest.raises(RuntimeError, match="RAG pipeline creation failed"):
            build_rag_with_evaluation_pipeline()
        
        # Verify error logging
        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[1]
        assert error_call_args["error"] == "RAG pipeline creation failed"
        assert error_call_args["error_type"] == "RuntimeError"
        assert error_call_args["component"] == "baseline_pipeline"


class TestRunBaselineEvaluationPipeline:
    """Test cases for baseline evaluation pipeline execution."""

    def test_run_baseline_evaluation_pipeline_with_valid_inputs(self):
        """Test baseline evaluation pipeline execution with valid inputs."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_single_result = {
            "retriever": {"documents": [Mock()]},
            "generator": {"replies": ["Test answer"]},
            "faithfulness_eval": {"score": 0.85},
            "context_eval": {"score": 0.92}
        }
        mock_pipeline.run.return_value = mock_single_result
        
        test_queries = ["What is Python?", "Who created it?"]
        ground_truth_answers = ["Programming language", "Guido van Rossum"]
        ground_truth_contexts = [["Python info"], ["Creator info"]]
        
        with patch('src.evaluation.pipelines.baseline_pipeline.get_logger'):
            # ACT
            result = run_baseline_evaluation_pipeline(
                mock_pipeline,
                test_queries,
                ground_truth_answers,
                ground_truth_contexts
            )
        
        # ASSERT
        assert "individual_results" in result
        assert "summary_metrics" in result
        assert len(result["individual_results"]) == len(test_queries)
        
        # Verify pipeline was called for each query
        assert mock_pipeline.run.call_count == len(test_queries)

    def test_run_baseline_evaluation_pipeline_handles_individual_query_errors(self):
        """Test that pipeline handles errors in individual query processing."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        # First query succeeds, second fails
        mock_pipeline.run.side_effect = [
            {"retriever": {"documents": []}, "generator": {"replies": ["Answer"]}},
            RuntimeError("Query processing failed")
        ]
        
        test_queries = ["Good query", "Bad query"]
        ground_truth_answers = ["Answer 1", "Answer 2"]
        ground_truth_contexts = [["Context 1"], ["Context 2"]]
        
        with patch('src.evaluation.pipelines.baseline_pipeline.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # ACT
            result = run_baseline_evaluation_pipeline(
                mock_pipeline,
                test_queries,
                ground_truth_answers,
                ground_truth_contexts
            )
        
        # ASSERT
        assert len(result["individual_results"]) == 1  # Only successful query
        
        # Verify error was logged
        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[1]
        assert error_call_args["query"] == "Bad query"
        assert "Query processing failed" in error_call_args["error"]

    def test_run_baseline_evaluation_pipeline_with_empty_inputs(self):
        """Test baseline evaluation pipeline with empty input lists."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        
        with patch('src.evaluation.pipelines.baseline_pipeline.get_logger'):
            # ACT
            result = run_baseline_evaluation_pipeline(mock_pipeline, [], [], [])
        
        # ASSERT
        assert result["individual_results"] == []
        assert "summary_metrics" in result
        mock_pipeline.run.assert_not_called()

    def test_run_baseline_evaluation_pipeline_with_mismatched_input_lengths(self):
        """Test pipeline handles mismatched input list lengths gracefully."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.run.return_value = {"generator": {"replies": ["Answer"]}}
        
        test_queries = ["Query 1", "Query 2"]
        ground_truth_answers = ["Answer 1"]  # Shorter list
        ground_truth_contexts = [["Context 1"]]  # Shorter list
        
        with patch('src.evaluation.pipelines.baseline_pipeline.get_logger'):
            # ACT
            result = run_baseline_evaluation_pipeline(
                mock_pipeline,
                test_queries,
                ground_truth_answers,
                ground_truth_contexts
            )
        
        # ASSERT
        # Should handle queries even when ground truth is shorter
        assert len(result["individual_results"]) == len(test_queries)


class TestCalculateSummaryMetrics:
    """Test cases for summary metrics calculation."""

    def test_calculate_summary_metrics_with_valid_results(self):
        """Test summary metrics calculation with valid evaluation results."""
        # ARRANGE
        sample_results = [
            {
                "faithfulness_eval": {"score": 0.8},
                "context_eval": {"score": 0.9},
                "generator": {"replies": ["Answer 1"]}
            },
            {
                "faithfulness_eval": {"score": 0.7},
                "context_eval": {"score": 0.85},
                "generator": {"replies": ["Answer 2"]}
            },
            None  # Simulate a failed evaluation
        ]
        
        # ACT
        summary = _calculate_summary_metrics(sample_results)
        
        # ASSERT
        assert isinstance(summary, dict)
        assert summary["total_queries"] == 3
        assert summary["successful_evaluations"] == 2  # Excluding None
        
        # Verify placeholder values (current implementation)
        assert "avg_faithfulness" in summary
        assert "avg_context_relevance" in summary

    def test_calculate_summary_metrics_with_empty_results(self):
        """Test summary metrics calculation with empty results list."""
        # ARRANGE & ACT
        summary = _calculate_summary_metrics([])
        
        # ASSERT
        assert summary == {}

    def test_calculate_summary_metrics_with_all_failed_results(self):
        """Test summary metrics calculation when all evaluations failed."""
        # ARRANGE
        failed_results = [None, None, None]
        
        # ACT
        summary = _calculate_summary_metrics(failed_results)
        
        # ASSERT
        assert summary["total_queries"] == 3
        assert summary["successful_evaluations"] == 0

    def test_calculate_summary_metrics_structure(self):
        """Test that summary metrics have expected structure and types."""
        # ARRANGE
        sample_results = [{"test": "data"}]
        
        # ACT
        summary = _calculate_summary_metrics(sample_results)
        
        # ASSERT
        # Verify all expected keys are present
        expected_keys = {"avg_faithfulness", "avg_context_relevance", "total_queries", "successful_evaluations"}
        assert set(summary.keys()) == expected_keys
        
        # Verify types
        assert isinstance(summary["total_queries"], int)
        assert isinstance(summary["successful_evaluations"], int)
        assert isinstance(summary["avg_faithfulness"], (int, float))
        assert isinstance(summary["avg_context_relevance"], (int, float))


class TestBaselinePipelineIntegration:
    """Integration tests for baseline pipeline components."""

    @patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline')
    def test_baseline_pipeline_component_connectivity(self, mock_build_rag):
        """Test that baseline pipeline properly connects RAG and evaluation components."""
        # ARRANGE
        mock_rag_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["retriever", "generator", "faithfulness_eval", "context_eval"]
        mock_rag_pipeline.graph = mock_graph
        mock_build_rag.return_value = mock_rag_pipeline
        
        # ACT
        pipeline = build_rag_with_evaluation_pipeline()
        
        # ASSERT
        # Verify evaluation components are added
        add_component_calls = [call[0] for call in mock_rag_pipeline.add_component.call_args_list]
        component_names = [call[0] for call in add_component_calls]
        assert "faithfulness_eval" in component_names
        assert "context_eval" in component_names

    def test_baseline_pipeline_evaluation_disabled_maintains_rag_functionality(self):
        """Test that disabling evaluation still maintains core RAG functionality."""
        # ARRANGE & ACT
        with patch('src.evaluation.pipelines.baseline_pipeline.build_rag_pipeline') as mock_build_rag:
            mock_rag_pipeline = Mock(spec=Pipeline)
            mock_build_rag.return_value = mock_rag_pipeline
            
            pipeline = build_rag_with_evaluation_pipeline(enable_evaluation=False)
        
        # ASSERT
        assert pipeline == mock_rag_pipeline
        # Verify original RAG pipeline is returned unchanged
        mock_rag_pipeline.add_component.assert_not_called()
        mock_rag_pipeline.connect.assert_not_called()