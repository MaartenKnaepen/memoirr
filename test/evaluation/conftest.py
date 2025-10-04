"""Pytest configuration and fixtures for evaluation tests."""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import pandas as pd

from haystack import Pipeline, Document
from haystack.components.evaluators import (
    FaithfulnessEvaluator,
    ContextRelevanceEvaluator,
    AnswerExactMatchEvaluator,
    DocumentRecallEvaluator,
    LLMEvaluator
)

from src.evaluation.test_data.ground_truth_builder import EvaluationDataPoint


@pytest.fixture
def sample_questions():
    """Sample questions for testing evaluators."""
    return [
        "Who created the Python language?",
        "What did Gandalf say about the ring?",
        "Where is Mordor located?"
    ]


@pytest.fixture
def sample_contexts():
    """Sample contexts for testing evaluators."""
    return [
        ["Python, created by Guido van Rossum in the late 1980s, is a high-level programming language."],
        ["Gandalf warned about the ring's power and its corrupting influence."],
        ["Mordor is a dark land in the southeast of Middle-earth, surrounded by mountains."]
    ]


@pytest.fixture
def sample_predicted_answers():
    """Sample predicted answers for testing evaluators."""
    return [
        "Python was created by Guido van Rossum.",
        "Gandalf said the ring was dangerous.",
        "Mordor is located in Middle-earth."
    ]


@pytest.fixture
def sample_ground_truth_answers():
    """Sample ground truth answers for testing evaluators."""
    return [
        "Guido van Rossum created Python in the late 1980s.",
        "Gandalf warned about the ring's power.",
        "Mordor is in the southeast of Middle-earth."
    ]


@pytest.fixture
def sample_documents():
    """Sample Haystack documents for testing."""
    return [
        Document(
            content="Python, created by Guido van Rossum in the late 1980s",
            meta={"doc_id": "python_doc_1", "score": 0.95}
        ),
        Document(
            content="Gandalf warned about the ring's corrupting power",
            meta={"doc_id": "gandalf_doc_1", "score": 0.88}
        ),
        Document(
            content="Mordor is a dark realm in Middle-earth",
            meta={"doc_id": "mordor_doc_1", "score": 0.92}
        )
    ]


@pytest.fixture
def sample_evaluation_data_points():
    """Sample EvaluationDataPoint objects for testing."""
    return [
        EvaluationDataPoint(
            query="Who created Python?",
            expected_answer="Guido van Rossum",
            relevant_document_ids=["python_doc_1"],
            ground_truth_contexts=["Python, created by Guido van Rossum"],
            evaluation_type="exact_match",
            metadata={"source": "test_data", "difficulty": "easy"}
        ),
        EvaluationDataPoint(
            query="What did Gandalf say about the ring?",
            expected_answer="The ring is dangerous",
            relevant_document_ids=["gandalf_doc_1"],
            ground_truth_contexts=["Gandalf warned about the ring's power"],
            evaluation_type="faithfulness",
            metadata={"source": "test_data", "difficulty": "medium"}
        )
    ]


@pytest.fixture
def mock_faithfulness_evaluator():
    """Mock FaithfulnessEvaluator for testing."""
    mock_evaluator = Mock(spec=FaithfulnessEvaluator)
    mock_evaluator.run.return_value = {
        "score": 0.85,
        "individual_scores": [0.9, 0.8],
        "results": [
            {"statements": ["Test statement"], "statement_scores": [1], "score": 0.9},
            {"statements": ["Another statement"], "statement_scores": [0], "score": 0.8}
        ]
    }
    return mock_evaluator


@pytest.fixture
def mock_context_relevance_evaluator():
    """Mock ContextRelevanceEvaluator for testing."""
    mock_evaluator = Mock(spec=ContextRelevanceEvaluator)
    mock_evaluator.run.return_value = {
        "score": 0.92,
        "individual_scores": [1.0, 0.84],
        "results": [
            {"statements": ["Relevant statement"], "statement_scores": [1], "score": 1.0},
            {"statements": ["Partially relevant"], "statement_scores": [1], "score": 0.84}
        ]
    }
    return mock_evaluator


@pytest.fixture
def mock_exact_match_evaluator():
    """Mock AnswerExactMatchEvaluator for testing."""
    mock_evaluator = Mock(spec=AnswerExactMatchEvaluator)
    mock_evaluator.run.return_value = {
        "score": 0.5,
        "individual_scores": [1, 0]
    }
    return mock_evaluator


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes.return_value = ["retriever", "generator"]
    mock_pipeline.run.return_value = {
        "retriever": {
            "documents": [
                Document(content="Sample retrieved content", meta={"score": 0.9})
            ]
        },
        "generator": {
            "replies": ["Sample generated answer"],
            "meta": [{"model": "test_model"}]
        }
    }
    return mock_pipeline


@pytest.fixture
def sample_metrics_dataframe():
    """Sample metrics DataFrame for testing visualization."""
    return pd.DataFrame([
        {
            "metric_name": "faithfulness",
            "value": 0.85,
            "timestamp": "2024-01-15T10:00:00",
            "evaluation_type": "faithfulness",
            "status": "completed"
        },
        {
            "metric_name": "context_relevance",
            "value": 0.92,
            "timestamp": "2024-01-15T10:00:00",
            "evaluation_type": "context_relevance",
            "status": "completed"
        },
        {
            "metric_name": "exact_match",
            "value": 0.5,
            "timestamp": "2024-01-15T10:00:00",
            "evaluation_type": "exact_match",
            "status": "completed"
        }
    ])


@pytest.fixture
def sample_baseline_results():
    """Sample baseline evaluation results for testing."""
    return {
        "faithfulness": 0.85,
        "context_relevance": 0.92,
        "exact_match": 0.5,
        "doc_recall": 0.8,
        "doc_mrr": 0.75,
        "avg_latency_ms": 1250.0,
        "p95_latency_ms": 1800.0,
        "p99_latency_ms": 2200.0
    }


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()
    mock_client.scroll.return_value = (
        [
            {
                "id": "doc_1",
                "payload": {
                    "text": "Sample text from LOTR",
                    "start_ms": 12000,
                    "end_ms": 15000,
                    "token_count": 10
                }
            },
            {
                "id": "doc_2", 
                "payload": {
                    "text": "Another sample text",
                    "start_ms": 20000,
                    "end_ms": 23000,
                    "token_count": 8
                }
            }
        ],
        None  # next_page_offset
    )
    return mock_client


@pytest.fixture
def temporary_output_dir(tmp_path):
    """Temporary directory for testing file outputs."""
    return str(tmp_path / "evaluation_output")


@pytest.fixture(autouse=True)
def mock_groq_api_key():
    """Mock GROQ_API_KEY environment variable for all tests."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_groq_api_key"}):
        yield


@pytest.fixture(autouse=True)
def mock_haystack_evaluators():
    """Mock all Haystack evaluators to avoid API calls during testing."""
    with patch('src.evaluation.haystack_evaluator.FaithfulnessEvaluator') as mock_faithfulness, \
         patch('src.evaluation.haystack_evaluator.ContextRelevanceEvaluator') as mock_context, \
         patch('src.evaluation.haystack_evaluator.AnswerExactMatchEvaluator') as mock_exact, \
         patch('src.evaluation.haystack_evaluator.DocumentRecallEvaluator') as mock_recall, \
         patch('src.evaluation.haystack_evaluator.DocumentMRREvaluator') as mock_mrr, \
         patch('src.evaluation.pipelines.evaluation_pipeline.FaithfulnessEvaluator') as mock_pipe_faith, \
         patch('src.evaluation.pipelines.evaluation_pipeline.ContextRelevanceEvaluator') as mock_pipe_context, \
         patch('src.evaluation.pipelines.evaluation_pipeline.AnswerExactMatchEvaluator') as mock_pipe_exact, \
         patch('src.evaluation.pipelines.evaluation_pipeline.DocumentRecallEvaluator') as mock_pipe_recall, \
         patch('src.evaluation.pipelines.evaluation_pipeline.DocumentMRREvaluator') as mock_pipe_mrr, \
         patch('src.evaluation.pipelines.baseline_pipeline.FaithfulnessEvaluator') as mock_base_faith, \
         patch('src.evaluation.pipelines.baseline_pipeline.ContextRelevanceEvaluator') as mock_base_context:
        
        # Configure all mocks to return Mock instances
        for mock_eval in [mock_faithfulness, mock_context, mock_exact, mock_recall, mock_mrr,
                         mock_pipe_faith, mock_pipe_context, mock_pipe_exact, mock_pipe_recall, mock_pipe_mrr,
                         mock_base_faith, mock_base_context]:
            mock_eval.return_value = Mock()
        
        yield {
            'faithfulness': mock_faithfulness,
            'context_relevance': mock_context,
            'exact_match': mock_exact,
            'doc_recall': mock_recall,
            'doc_mrr': mock_mrr
        }