"""Haystack pipeline with native evaluation components.

This module builds evaluation-focused pipelines using Haystack's built-in
evaluator components for measuring RAG system performance.
"""

from typing import Optional, Dict, Any
import os

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
        
        # Add statistical evaluators (no API key required)
        eval_pipeline.add_component("exact_match", AnswerExactMatchEvaluator())
        eval_pipeline.add_component("doc_recall", DocumentRecallEvaluator())
        eval_pipeline.add_component("doc_mrr", DocumentMRREvaluator())
        
        # Add LLM-based evaluators with proper API configuration
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            # Configure LLM-based evaluators to use Groq API
            from haystack.components.generators import OpenAIGenerator
            from haystack.utils import Secret
            
            # Create Groq-compatible chat generator
            groq_generator = OpenAIGenerator(
                api_key=Secret.from_env_var("GROQ_API_KEY"),
                api_base_url="https://api.groq.com/openai/v1",
                model="llama3-8b-8192",
                generation_kwargs={"response_format": {"type": "json_object"}}
            )
            
            # Add LLM-based evaluators with Groq generator
            eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator(chat_generator=groq_generator))
            eval_pipeline.add_component("context_relevance", ContextRelevanceEvaluator(chat_generator=groq_generator))
        else:
            logger.warning(
                "GROQ_API_KEY not found, skipping LLM-based evaluators",
                component="evaluation_pipeline"
            )
        
        # Apply configuration overrides if provided
        if evaluator_config:
            eval_pipeline = _apply_evaluator_config(eval_pipeline, evaluator_config)
        
        # Add custom LLM evaluator for missing features
        if llm_api_key:
            missing_features_evaluator = _create_missing_features_evaluator(llm_api_key)
            eval_pipeline.add_component("missing_features", missing_features_evaluator)
        
        # Validate pipeline construction
        if not _validate_evaluation_pipeline(eval_pipeline):
            raise ValueError("Evaluation pipeline validation failed")
        
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


def _create_missing_features_evaluator(llm_api_key: str) -> LLMEvaluator:
    """Create LLM evaluator specifically for missing feature detection.
    
    Args:
        llm_api_key: API key for LLM-based evaluation
        
    Returns:
        Configured LLMEvaluator for missing features
    """
    from typing import List
    
    return LLMEvaluator(
        instructions=(
            "Evaluate if the RAG system response includes:\n"
            "1. Speaker attribution (who said the quote)\n"
            "2. Hybrid search results (BM25 + semantic)\n" 
            "3. Conversation threading (dialogue context)\n"
            "Return specific scores for each feature (0-1) and explanations."
        ),
        inputs=[("responses", List[str]), ("queries", List[str])],
        outputs=["speaker_score", "hybrid_score", "threading_score", "explanation"],
        examples=[
            {
                "inputs": {
                    "responses": ["Gandalf said this quote but I don't know who the speaker is"],
                    "queries": ["Who said 'One does not simply walk into Mordor'?"]
                },
                "outputs": {
                    "speaker_score": 0,
                    "hybrid_score": 0,
                    "threading_score": 0,
                    "explanation": "No speaker attribution, standard semantic search, no conversation context"
                }
            },
            {
                "inputs": {
                    "responses": ["Speaker: Boromir said 'One does not simply walk into Mordor' during the Council of Elrond"],
                    "queries": ["Who said 'One does not simply walk into Mordor'?"]
                },
                "outputs": {
                    "speaker_score": 1,
                    "hybrid_score": 0,
                    "threading_score": 1,
                    "explanation": "Clear speaker attribution and conversation context, but standard search"
                }
            }
        ]
    )


def _apply_evaluator_config(pipeline: Pipeline, config: Dict[str, Any]) -> Pipeline:
    """Apply configuration overrides to evaluator components.
    
    Args:
        pipeline: Pipeline with evaluator components
        config: Configuration overrides for evaluators
        
    Returns:
        Pipeline with updated component configurations
    """
    logger = get_logger(__name__)
    
    try:
        # Handle LLM-based evaluator configurations
        if "faithfulness" in config:
            faithfulness_config = config["faithfulness"]
            # Replace component with configured version
            pipeline.remove_component("faithfulness")
            pipeline.add_component("faithfulness", FaithfulnessEvaluator(**faithfulness_config))
            
        if "context_relevance" in config:
            context_config = config["context_relevance"]
            pipeline.remove_component("context_relevance")
            pipeline.add_component("context_relevance", ContextRelevanceEvaluator(**context_config))
            
        logger.info(
            "Applied evaluator configuration overrides",
            configured_components=list(config.keys()),
            component="evaluation_pipeline"
        )
        
        return pipeline
        
    except Exception as e:
        logger.error(
            "Failed to apply evaluator configuration",
            error=str(e),
            error_type=type(e).__name__,
            component="evaluation_pipeline"
        )
        raise


def _validate_evaluation_pipeline(pipeline: Pipeline) -> bool:
    """Validate that evaluation pipeline is properly constructed.
    
    Args:
        pipeline: Pipeline to validate
        
    Returns:
        True if pipeline is valid, False otherwise
    """
    logger = get_logger(__name__)
    
    # Core required components (statistical evaluators that don't need API keys)
    core_required_components = [
        "exact_match", "doc_recall", "doc_mrr"
    ]
    
    # Optional LLM-based components (require API keys)
    optional_llm_components = [
        "faithfulness", "context_relevance"
    ]
    
    pipeline_components = list(pipeline.graph.nodes())
    
    # Check core components
    for component in core_required_components:
        if component not in pipeline_components:
            logger.error(
                "Missing required evaluator component",
                missing_component=component,
                available_components=pipeline_components,
                component="evaluation_pipeline"
            )
            return False
    
    # Check if LLM components are present (log info, don't fail validation)
    missing_llm_components = [comp for comp in optional_llm_components if comp not in pipeline_components]
    if missing_llm_components:
        logger.info(
            "LLM-based evaluators not available (API key required)",
            missing_llm_components=missing_llm_components,
            available_components=pipeline_components,
            component="evaluation_pipeline"
        )
    
    logger.info(
        "Evaluation pipeline validation successful",
        total_components=len(pipeline_components),
        core_required_components=len(core_required_components),
        llm_components_available=len([c for c in optional_llm_components if c in pipeline_components]),
        component="evaluation_pipeline"
    )
    
    return True


def run_document_recall_evaluation(
    pipeline: Pipeline,
    ground_truth_documents: list[list],
    retrieved_documents: list[list]
) -> Dict[str, Any]:
    """Run document recall evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with document recall component
        ground_truth_documents: List of ground truth document lists for each query
        retrieved_documents: List of retrieved document lists for each query
        
    Returns:
        Document recall evaluation results
    """
    return pipeline.run({
        "doc_recall": {
            "ground_truth_documents": ground_truth_documents,
            "retrieved_documents": retrieved_documents
        }
    })


def run_document_mrr_evaluation(
    pipeline: Pipeline,
    ground_truth_documents: list[list],
    retrieved_documents: list[list]
) -> Dict[str, Any]:
    """Run document MRR evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with document MRR component
        ground_truth_documents: List of ground truth document lists for each query
        retrieved_documents: List of retrieved document lists for each query
        
    Returns:
        Document MRR evaluation results
    """
    return pipeline.run({
        "doc_mrr": {
            "ground_truth_documents": ground_truth_documents,
            "retrieved_documents": retrieved_documents
        }
    })


def run_missing_features_evaluation(
    pipeline: Pipeline,
    responses: list[str],
    queries: list[str]
) -> Dict[str, Any]:
    """Run missing features evaluation using the pipeline.
    
    Args:
        pipeline: Evaluation pipeline with missing features component
        responses: List of RAG system responses
        queries: List of queries that generated the responses
        
    Returns:
        Missing features evaluation results
    """
    if "missing_features" not in pipeline.graph.nodes():
        raise ValueError("Pipeline does not contain missing_features component")
        
    return pipeline.run({
        "missing_features": {
            "responses": responses,
            "queries": queries
        }
    })