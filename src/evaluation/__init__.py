"""Haystack-native RAG evaluation framework.

This module provides comprehensive evaluation capabilities for RAG systems using
Haystack's built-in evaluator components. It includes baseline measurement,
missing feature tracking, and visualization tools.

Key Components:
- HaystackRAGEvaluator: Main evaluation orchestrator
- Evaluation pipelines: Combined RAG + evaluation workflows
- Ground truth generation: Extract test data from Qdrant
- Visualization: DataFrames and charts for analysis
"""

from src.evaluation.haystack_evaluator import HaystackRAGEvaluator

__all__ = ["HaystackRAGEvaluator"]