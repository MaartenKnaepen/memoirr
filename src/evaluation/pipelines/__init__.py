"""Evaluation pipeline components for Haystack RAG evaluation.

This module provides pipeline components that combine RAG functionality
with Haystack's native evaluation components for comprehensive testing.
"""

from src.evaluation.pipelines.evaluation_pipeline import build_evaluation_pipeline
from src.evaluation.pipelines.baseline_pipeline import build_rag_with_evaluation_pipeline

__all__ = ["build_evaluation_pipeline", "build_rag_with_evaluation_pipeline"]