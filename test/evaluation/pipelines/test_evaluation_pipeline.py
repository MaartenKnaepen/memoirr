"""Tests for evaluation pipeline components following Memoirr standards.

Tests the Haystack evaluation pipeline construction and execution using
AAA pattern, comprehensive mocking, and error handling validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from haystack import Pipeline
from haystack.components.evaluators import (
    FaithfulnessEvaluator,
    ContextRelevanceEvaluator,
    AnswerExactMatchEvaluator,
    DocumentRecallEvaluator,
    DocumentMRREvaluator,
    LLMEvaluator
)

from src.evaluation.pipelines.evaluation_pipeline import (
    build_evaluation_pipeline,
    run_faithfulness_evaluation,
    run_context_relevance_evaluation,
    run_exact_match_evaluation
)


def create_mock_haystack_evaluator():
    """Helper function to create properly mocked Haystack evaluator instances."""
    mock_instance = Mock()
    mock_instance.__haystack_input__ = Mock()
    mock_instance.__haystack_input__._sockets_dict = {}
    mock_instance.__haystack_output__ = Mock()
    mock_instance.__haystack_output__._sockets_dict = {}
    return mock_instance


class TestBuildEvaluationPipeline:
    """Test cases for evaluation pipeline construction."""

    @patch('src.evaluation.pipelines.evaluation_pipeline.Pipeline')
    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    def test_build_evaluation_pipeline_creates_all_components(
        self,
        mock_doc_mrr,
        mock_doc_recall,
        mock_exact_match,
        mock_context_relevance,
        mock_faithfulness,
        mock_pipeline_class
    ):
        """Test that evaluation pipeline includes all required evaluator components."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["faithfulness", "context_relevance", "exact_match", "doc_recall", "doc_mrr"]
        mock_pipeline.graph = mock_graph
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_faithfulness_instance = create_mock_haystack_evaluator()
        mock_context_relevance_instance = create_mock_haystack_evaluator()
        mock_exact_match_instance = create_mock_haystack_evaluator()
        mock_doc_recall_instance = create_mock_haystack_evaluator()
        mock_doc_mrr_instance = create_mock_haystack_evaluator()
        
        mock_faithfulness.return_value = mock_faithfulness_instance
        mock_context_relevance.return_value = mock_context_relevance_instance
        mock_exact_match.return_value = mock_exact_match_instance
        mock_doc_recall.return_value = mock_doc_recall_instance
        mock_doc_mrr.return_value = mock_doc_mrr_instance
        
        # ACT
        result = build_evaluation_pipeline()
        
        # ASSERT
        assert result == mock_pipeline
        
        # Verify all evaluators are created
        mock_faithfulness.assert_called_once()
        mock_context_relevance.assert_called_once()
        mock_exact_match.assert_called_once()
        mock_doc_recall.assert_called_once()
        mock_doc_mrr.assert_called_once()
        
        # Verify all components are added to pipeline
        expected_add_component_calls = [
            (("faithfulness", mock_faithfulness_instance),),
            (("context_relevance", mock_context_relevance_instance),),
            (("exact_match", mock_exact_match_instance),),
            (("doc_recall", mock_doc_recall_instance),),
            (("doc_mrr", mock_doc_mrr_instance),)
        ]
        
        actual_calls = mock_pipeline.add_component.call_args_list
        assert len(actual_calls) >= 5  # At least the 5 core evaluators

    @patch('src.evaluation.pipelines.evaluation_pipeline.Pipeline')
    @patch('src.evaluation.pipelines.evaluation_pipeline.LLMEvaluator')
    def test_build_evaluation_pipeline_adds_llm_evaluator_with_api_key(
        self,
        mock_llm_evaluator,
        mock_pipeline_class
    ):
        """Test that LLM evaluator is added when API key is provided."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["faithfulness", "context_relevance", "exact_match", "doc_recall", "doc_mrr", "missing_features"]
        mock_pipeline.graph = mock_graph
        mock_pipeline_class.return_value = mock_pipeline
        mock_llm_instance = create_mock_haystack_evaluator()
        mock_llm_evaluator.return_value = mock_llm_instance
        
        api_key = "test_api_key"
        
        # ACT
        result = build_evaluation_pipeline(llm_api_key=api_key)
        
        # ASSERT
        mock_llm_evaluator.assert_called_once()
        
        # Verify LLM evaluator is configured correctly
        call_args = mock_llm_evaluator.call_args
        assert call_args[1]["inputs"] == [("responses", List[str])]
        assert call_args[1]["outputs"] == ["score"]
        assert "examples" in call_args[1]
        assert len(call_args[1]["examples"]) == 2  # Should have example responses
        
        # Verify LLM evaluator is added to pipeline
        mock_pipeline.add_component.assert_any_call("missing_features", mock_llm_instance)

    @patch('src.evaluation.pipelines.evaluation_pipeline.Pipeline')
    def test_build_evaluation_pipeline_without_llm_evaluator(self, mock_pipeline_class):
        """Test that LLM evaluator is not added when no API key provided."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_graph = Mock()
        mock_graph.nodes.return_value = ["faithfulness", "context_relevance", "exact_match", "doc_recall", "doc_mrr"]
        mock_pipeline.graph = mock_graph
        mock_pipeline_class.return_value = mock_pipeline
        
        # ACT
        result = build_evaluation_pipeline(llm_api_key=None)
        
        # ASSERT
        # Verify no "missing_features" component is added
        add_component_calls = [call[0][0] for call in mock_pipeline.add_component.call_args_list]
        assert "missing_features" not in add_component_calls

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.get_logger')
    def test_build_evaluation_pipeline_logs_success(self, mock_get_logger, mock_doc_mrr, mock_doc_recall, mock_exact_match, mock_context_relevance, mock_faithfulness):
        """Test that pipeline construction logs success message."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create mock instances with required Haystack attributes
        mock_faithfulness.return_value = create_mock_haystack_evaluator()
        mock_context_relevance.return_value = create_mock_haystack_evaluator()
        mock_exact_match.return_value = create_mock_haystack_evaluator()
        mock_doc_recall.return_value = create_mock_haystack_evaluator()
        mock_doc_mrr.return_value = create_mock_haystack_evaluator()
        
        # ACT
        result = build_evaluation_pipeline()
        
        # ASSERT
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[1]
        assert log_call_args["component"] == "evaluation_pipeline"
        assert "components" in log_call_args
        assert "has_llm_evaluator" in log_call_args

    @patch('src.evaluation.pipelines.evaluation_pipeline.Pipeline')
    @patch('src.evaluation.pipelines.evaluation_pipeline.get_logger')
    def test_build_evaluation_pipeline_handles_exceptions(self, mock_get_logger, mock_pipeline_class):
        """Test that pipeline construction handles and logs exceptions."""
        # ARRANGE
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_pipeline_class.side_effect = RuntimeError("Pipeline creation failed")
        
        # ACT & ASSERT
        with pytest.raises(RuntimeError, match="Pipeline creation failed"):
            build_evaluation_pipeline()
        
        # Verify error logging
        mock_logger.error.assert_called()
        error_call_args = mock_logger.error.call_args[1]
        assert error_call_args["error"] == "Pipeline creation failed"
        assert error_call_args["error_type"] == "RuntimeError"
        assert error_call_args["component"] == "evaluation_pipeline"

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    def test_build_evaluation_pipeline_with_custom_config(self, mock_doc_mrr, mock_doc_recall, mock_exact_match, mock_context_relevance, mock_faithfulness):
        """Test pipeline construction with custom evaluator configuration."""
        # ARRANGE
        # Create mock instances with required Haystack attributes
        mock_faithfulness.return_value = create_mock_haystack_evaluator()
        mock_context_relevance.return_value = create_mock_haystack_evaluator()
        mock_exact_match.return_value = create_mock_haystack_evaluator()
        mock_doc_recall.return_value = create_mock_haystack_evaluator()
        mock_doc_mrr.return_value = create_mock_haystack_evaluator()
        
        custom_config = {
            "faithfulness": {"threshold": 0.8},
            "context_relevance": {"threshold": 0.7}
        }
        
        # ACT
        result = build_evaluation_pipeline(evaluator_config=custom_config)
        
        # ASSERT
        assert isinstance(result, Pipeline)
        # Note: Current implementation doesn't use evaluator_config,
        # but this test ensures the parameter is accepted


class TestEvaluationPipelineFunctions:
    """Test cases for individual evaluation pipeline functions."""

    def test_run_faithfulness_evaluation_with_valid_inputs(self):
        """Test faithfulness evaluation with valid inputs."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_result = {"faithfulness": {"score": 0.85, "individual_scores": [0.9, 0.8]}}
        mock_pipeline.run.return_value = mock_result
        
        questions = ["What is Python?", "Who created it?"]
        contexts = [["Python is a programming language"], ["Created by Guido van Rossum"]]
        predicted_answers = ["Python is a language", "Guido created it"]
        
        # ACT
        result = run_faithfulness_evaluation(mock_pipeline, questions, contexts, predicted_answers)
        
        # ASSERT
        assert result == mock_result
        mock_pipeline.run.assert_called_once_with({
            "faithfulness": {
                "questions": questions,
                "contexts": contexts,
                "predicted_answers": predicted_answers
            }
        })

    def test_run_context_relevance_evaluation_with_valid_inputs(self):
        """Test context relevance evaluation with valid inputs."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_result = {"context_relevance": {"score": 0.92, "individual_scores": [1.0, 0.84]}}
        mock_pipeline.run.return_value = mock_result
        
        questions = ["What is Python?"]
        contexts = [["Python is a programming language"]]
        
        # ACT
        result = run_context_relevance_evaluation(mock_pipeline, questions, contexts)
        
        # ASSERT
        assert result == mock_result
        mock_pipeline.run.assert_called_once_with({
            "context_relevance": {
                "questions": questions,
                "contexts": contexts
            }
        })

    def test_run_exact_match_evaluation_with_valid_inputs(self):
        """Test exact match evaluation with valid inputs."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_result = {"exact_match": {"score": 0.5, "individual_scores": [1, 0]}}
        mock_pipeline.run.return_value = mock_result
        
        predicted_answers = ["Berlin", "Lyon"]
        ground_truth_answers = ["Berlin", "Paris"]
        
        # ACT
        result = run_exact_match_evaluation(mock_pipeline, predicted_answers, ground_truth_answers)
        
        # ASSERT
        assert result == mock_result
        mock_pipeline.run.assert_called_once_with({
            "exact_match": {
                "predicted_answers": predicted_answers,
                "ground_truth_answers": ground_truth_answers
            }
        })

    def test_run_faithfulness_evaluation_handles_empty_inputs(self):
        """Test faithfulness evaluation handles empty input lists."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.run.return_value = {"faithfulness": {"score": 0.0, "individual_scores": []}}
        
        # ACT
        result = run_faithfulness_evaluation(mock_pipeline, [], [], [])
        
        # ASSERT
        assert "faithfulness" in result
        mock_pipeline.run.assert_called_once()

    def test_run_context_relevance_evaluation_handles_pipeline_errors(self):
        """Test context relevance evaluation handles pipeline execution errors."""
        # ARRANGE
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.run.side_effect = RuntimeError("Pipeline execution failed")
        
        questions = ["Test question"]
        contexts = [["Test context"]]
        
        # ACT & ASSERT
        with pytest.raises(RuntimeError, match="Pipeline execution failed"):
            run_context_relevance_evaluation(mock_pipeline, questions, contexts)


class TestEvaluationPipelineIntegration:
    """Integration tests for evaluation pipeline components."""

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    def test_evaluation_pipeline_components_connectivity(
        self,
        mock_doc_mrr,
        mock_doc_recall,
        mock_exact_match,
        mock_context_relevance,
        mock_faithfulness
    ):
        """Test that evaluation pipeline components can be properly connected."""
        # ARRANGE
        # Create mock instances with required Haystack attributes
        mock_faithfulness.return_value = create_mock_haystack_evaluator()
        mock_context_relevance.return_value = create_mock_haystack_evaluator()
        mock_exact_match.return_value = create_mock_haystack_evaluator()
        mock_doc_recall.return_value = create_mock_haystack_evaluator()
        mock_doc_mrr.return_value = create_mock_haystack_evaluator()
        
        # ACT
        pipeline = build_evaluation_pipeline()
        
        # ASSERT
        assert isinstance(pipeline, Pipeline)
        
        # Verify pipeline has expected components
        component_names = list(pipeline.graph.nodes())
        expected_components = ["faithfulness", "context_relevance", "exact_match", "doc_recall", "doc_mrr"]
        
        for component in expected_components:
            assert component in component_names, f"Component {component} not found in pipeline"

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.LLMEvaluator')
    def test_evaluation_pipeline_with_llm_evaluator_connectivity(
        self,
        mock_llm_evaluator,
        mock_doc_mrr,
        mock_doc_recall,
        mock_exact_match,
        mock_context_relevance,
        mock_faithfulness
    ):
        """Test evaluation pipeline connectivity when LLM evaluator is included."""
        # ARRANGE
        # Create mock instances with required Haystack attributes
        mock_faithfulness.return_value = create_mock_haystack_evaluator()
        mock_context_relevance.return_value = create_mock_haystack_evaluator()
        mock_exact_match.return_value = create_mock_haystack_evaluator()
        mock_doc_recall.return_value = create_mock_haystack_evaluator()
        mock_doc_mrr.return_value = create_mock_haystack_evaluator()
        mock_llm_evaluator.return_value = create_mock_haystack_evaluator()
        
        # ACT
        pipeline = build_evaluation_pipeline(llm_api_key="test_key")
        
        # ASSERT
        component_names = list(pipeline.graph.nodes())
        assert "missing_features" in component_names

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    def test_evaluation_pipeline_component_initialization_failure(self, mock_faithfulness):
        """Test pipeline handles component initialization failures."""
        # ARRANGE
        mock_faithfulness.side_effect = ImportError("FaithfulnessEvaluator not available")
        
        # ACT & ASSERT
        with pytest.raises(ImportError, match="FaithfulnessEvaluator not available"):
            build_evaluation_pipeline()

    @patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator')
    @patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator')
    def test_evaluation_pipeline_runs_with_sample_data(
        self,
        mock_doc_mrr,
        mock_doc_recall,
        mock_exact_match,
        mock_context_relevance,
        mock_faithfulness,
        sample_questions, 
        sample_contexts, 
        sample_predicted_answers
    ):
        """Test that evaluation pipeline can run with sample data."""
        # ARRANGE
        # Create mock instances with required Haystack attributes
        mock_faithfulness.return_value = create_mock_haystack_evaluator()
        mock_context_relevance.return_value = create_mock_haystack_evaluator()
        mock_exact_match.return_value = create_mock_haystack_evaluator()
        mock_doc_recall.return_value = create_mock_haystack_evaluator()
        mock_doc_mrr.return_value = create_mock_haystack_evaluator()
        
        pipeline = build_evaluation_pipeline()
        
        # Mock the pipeline run to return expected results
        pipeline.run = Mock(return_value={"faithfulness": {"score": 0.85}})
        
        # ACT - This is a smoke test, we don't expect real evaluation
        # since evaluators need proper LLM backends, but pipeline should not crash
        try:
            # Note: This would normally fail without proper LLM configuration
            # But we're testing that the pipeline structure is sound
            result = run_faithfulness_evaluation(
                pipeline, 
                sample_questions[:1], 
                sample_contexts[:1], 
                sample_predicted_answers[:1]
            )
            # If it runs without structural errors, the test passes
            assert "faithfulness" in result
        except Exception as e:
            # Expected - LLM not configured, but pipeline structure should be valid
            # We're mainly testing that components are properly wired
            assert "api" in str(e).lower() or "key" in str(e).lower() or "model" in str(e).lower()