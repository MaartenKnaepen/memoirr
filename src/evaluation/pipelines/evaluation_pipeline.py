"""Haystack pipeline with native evaluation components.

This module builds evaluation-focused pipelines using Haystack's built-in
evaluator components for measuring RAG system performance.
"""

from typing import Optional, Dict, Any

from haystack import Pipeline
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
    AnswerExactMatchEvaluator,
    DocumentRecallEvaluator,
    DocumentMRREvaluator,
    LLMEvaluator
)

from src.core.logging_config import get_logger


def build_evaluation_pipeline(
    llm_api_key: Optional[str] = None,
    evaluator_config: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """Build pipeline with RAG + Haystack evaluators.
    
    Creates a Haystack pipeline containing multiple evaluation components
    that can assess different aspects of RAG system performance.
    
    Args:
        llm_api_key: API key for LLM-based evaluators (Groq, OpenAI, etc.)
        evaluator_config: Configuration overrides for evaluator components
        
    Returns:
        Configured Haystack Pipeline with evaluation components
        
    Example:
        eval_pipeline = build_evaluation_pipeline()
        results = eval_pipeline.run({
            "faithfulness": {
                "questions": ["What did Gandalf say?"],
                "contexts": [["Gandalf spoke about the ring."]],
                "predicted_answers": ["Gandalf mentioned the ring."]
            }
        })
    """
    logger = get_logger(__name__)
    
    try:
        eval_pipeline = Pipeline()
        
        # Add evaluation components
        eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
        eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator())
        eval_pipeline.add_component("exact_match", AnswerExactMatchEvaluator())
        eval_pipeline.add_component("doc_recall", DocumentRecallEvaluator())
        eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
        
        # Add custom LLM evaluator for missing features
        if llm_api_key:
            from typing import List
            missing_features_evaluator = LLMEvaluator(
                instructions=(
                    "Evaluate if the RAG system response includes speaker attribution, "
                    "hybrid search results, or conversation threading. "
                    "Return a score from 0-1 and explain what features are missing."
                ),
                inputs=[("responses", List[str])],
                outputs=["score"],
                examples=[
                    {
                        "inputs": {"responses": "Gandalf said this quote but I don't know who the speaker is"},
                        "outputs": {"score": 0}
                    },
                    {
                        "inputs": {"responses": "Speaker: Gandalf said this quote"},
                        "outputs": {"score": 1}
                    }
                ]
            )
            eval_pipeline.add_component("missing_features", missing_features_evaluator)
        
        logger.info(
            "Evaluation pipeline built successfully",
            components=list(eval_pipeline.graph.nodes()),
            has_llm_evaluator=llm_api_key is not None,
            component="evaluation_pipeline"
        )
        
        return eval_pipeline
        
    except Exception as e:
        logger.error(
            "Failed to build evaluation pipeline",
            error=str(e),
            error_type=type(e).__name__,
            component="evaluation_pipeline"
        )
        raise


def run_faithfulness_evaluation(
    pipeline: Pipeline,
    questions: list[str],
    contexts: list[list[str]],
    predicted_answers: list[str]
) -> Dict[str, Any]:
    """Run faithfulness evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with faithfulness component
        questions: List of questions asked
        contexts: List of context lists for each question
        predicted_answers: List of predicted answers
        
    Returns:
        Faithfulness evaluation results
    """
    return pipeline.run({
        "faithfulness": {
            "questions": questions,
            "contexts": contexts,
            "predicted_answers": predicted_answers
        }
    })


def run_context_relevance_evaluation(
    pipeline: Pipeline,
    questions: list[str],
    contexts: list[list[str]]
) -> Dict[str, Any]:
    """Run context relevance evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with context relevance component
        questions: List of questions asked
        contexts: List of context lists for each question
        
    Returns:
        Context relevance evaluation results
    """
    return pipeline.run({
        "context_relevance": {
            "questions": questions,
            "contexts": contexts
        }
    })


def run_exact_match_evaluation(
    pipeline: Pipeline,
    predicted_answers: list[str],
    ground_truth_answers: list[str]
) -> Dict[str, Any]:
    """Run exact match evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with exact match component
        predicted_answers: List of predicted answers
        ground_truth_answers: List of ground truth answers
        
    Returns:
        Exact match evaluation results
    """
    return pipeline.run({
        "exact_match": {
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers
        }
    })