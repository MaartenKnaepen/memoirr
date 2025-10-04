"""Combined RAG + evaluation pipeline for baseline measurement.

This module creates pipelines that integrate existing RAG functionality
with evaluation components for end-to-end performance measurement.
"""

from typing import Optional, Dict, Any

from haystack import Pipeline
from haystack.components.evaluators import (
    FaithfulnessEvaluator,
    ContextRelevanceEvaluator
)

from src.core.logging_config import get_logger
from src.pipelines.rag_pipeline import build_rag_pipeline


def build_rag_with_evaluation_pipeline(
    retriever_config: Optional[Dict[str, Any]] = None,
    generator_config: Optional[Dict[str, Any]] = None,
    enable_evaluation: bool = True
) -> Pipeline:
    """Combine existing RAG with Haystack evaluators.
    
    Creates a pipeline that includes both RAG functionality and evaluation
    components, enabling end-to-end evaluation in a single pipeline run.
    
    Args:
        retriever_config: Configuration for QdrantRetriever component
        generator_config: Configuration for GroqGenerator component  
        enable_evaluation: Whether to include evaluation components
        
    Returns:
        Combined RAG + evaluation pipeline
        
    Example:
        pipeline = build_rag_with_evaluation_pipeline()
        results = pipeline.run({
            "query": "What did Gandalf say about the ring?",
            "ground_truth_answer": "Gandalf warned about the ring's power"
        })
        
        # Results include both RAG response and evaluation metrics
        answer = results["generator"]["replies"][0]
        faithfulness = results["faithfulness_eval"]["score"]
    """
    logger = get_logger(__name__)
    
    try:
        # Build base RAG pipeline
        rag_pipeline = build_rag_pipeline(
            retriever_config=retriever_config,
            generator_config=generator_config
        )
        
        if not enable_evaluation:
            return rag_pipeline
        
        # Add evaluation components
        rag_pipeline.add_component("faithfulness_eval", FaithfulnessEvaluator())
        rag_pipeline.add_component("context_eval", ContextRelevanceEvaluator())
        
        # Connect RAG outputs to evaluation inputs
        # Note: Exact connection syntax depends on your RAG pipeline structure
        # This will need to be updated based on actual component names
        
        # TODO: Add proper connections based on actual RAG pipeline structure
        # Example connections (to be implemented in Day 2):
        # rag_pipeline.connect("generator.replies", "faithfulness_eval.predicted_answers")
        # rag_pipeline.connect("retriever.documents", "context_eval.contexts")
        
        logger.info(
            "RAG + evaluation pipeline built successfully",
            total_components=len(list(rag_pipeline.graph.nodes())),
            evaluation_enabled=enable_evaluation,
            component="baseline_pipeline"
        )
        
        return rag_pipeline
        
    except Exception as e:
        logger.error(
            "Failed to build RAG + evaluation pipeline",
            error=str(e),
            error_type=type(e).__name__,
            component="baseline_pipeline"
        )
        raise


def run_baseline_evaluation_pipeline(
    pipeline: Pipeline,
    test_queries: list[str],
    ground_truth_answers: list[str],
    ground_truth_contexts: list[list[str]]
) -> Dict[str, Any]:
    """Run the combined pipeline for baseline evaluation.
    
    Args:
        pipeline: Combined RAG + evaluation pipeline
        test_queries: List of test questions
        ground_truth_answers: Expected answers for evaluation
        ground_truth_contexts: Expected contexts for evaluation
        
    Returns:
        Combined results including RAG responses and evaluation metrics
    """
    logger = get_logger(__name__)
    
    all_results = []
    
    for i, query in enumerate(test_queries):
        try:
            # Run pipeline for single query
            result = pipeline.run({
                "query": query,
                # Additional evaluation inputs
                "ground_truth_answer": ground_truth_answers[i] if i < len(ground_truth_answers) else None,
                "ground_truth_context": ground_truth_contexts[i] if i < len(ground_truth_contexts) else None
            })
            
            all_results.append(result)
            
        except Exception as e:
            logger.error(
                "Baseline evaluation failed for query",
                query=query,
                error=str(e),
                component="baseline_pipeline"
            )
            
    return {
        "individual_results": all_results,
        "summary_metrics": _calculate_summary_metrics(all_results)
    }


def _calculate_summary_metrics(results: list[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate summary metrics from individual evaluation results.
    
    Args:
        results: List of individual pipeline run results
        
    Returns:
        Summary statistics across all evaluations
    """
    if not results:
        return {}
    
    # TODO: Implement summary metric calculation
    # This will aggregate scores from individual evaluations
    
    return {
        "avg_faithfulness": 0.0,
        "avg_context_relevance": 0.0,
        "total_queries": len(results),
        "successful_evaluations": len([r for r in results if r is not None])
    }