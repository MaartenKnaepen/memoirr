"""Evaluation pipeline components for Haystack RAG evaluation.

This module provides pipeline components that combine RAG functionality
with Haystack's native evaluation components for comprehensive testing.
"""

from src.evaluation.pipelines.evaluation_pipeline import (
    build_evaluation_pipeline,
    run_faithfulness_evaluation,
    run_context_relevance_evaluation,
    run_exact_match_evaluation,
    run_document_recall_evaluation,
    run_document_mrr_evaluation,
    run_missing_features_evaluation
)
from src.evaluation.pipelines.baseline_pipeline import build_rag_with_evaluation_pipeline

__all__ = [
    "build_evaluation_pipeline",
    "build_rag_with_evaluation_pipeline",
    "run_faithfulness_evaluation",
    "run_context_relevance_evaluation", 
    "run_exact_match_evaluation",
    "run_document_recall_evaluation",
    "run_document_mrr_evaluation",
    "run_missing_features_evaluation"
]