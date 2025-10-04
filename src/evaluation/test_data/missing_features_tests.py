"""Tests for unimplemented RAG features using Haystack's LLMEvaluator.

This module creates failing tests that track features not yet implemented
in the RAG system, providing clear roadmap visibility and progress tracking.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta

from haystack.components.evaluators import LLMEvaluator

from src.core.logging_config import get_logger
from src.evaluation.test_data.ground_truth_builder import EvaluationDataPoint


def create_speaker_attribution_tests() -> List[EvaluationDataPoint]:
    """Create tests that will fail because speaker detection isn't implemented.
    
    These tests check whether the RAG system can identify who said specific
    quotes - a feature not yet implemented in the current system.
    
    Returns:
        List of speaker attribution test cases that should fail
        
    Example:
        tests = create_speaker_attribution_tests()
        # All tests should fail with current system lacking speaker detection
    """
    logger = get_logger(__name__)
    
    speaker_tests = [
        {
            "query": "Who said 'One does not simply walk into Mordor'?",
            "expected_answer": "Boromir",
            "context": "Boromir speaking during the Council of Elrond about the dangers of Mordor",
            "failure_reason": "No speaker attribution in current RAG system"
        },
        {
            "query": "Which character said 'My precious'?",
            "expected_answer": "Gollum",
            "context": "Gollum referring to the One Ring throughout the story",
            "failure_reason": "Speaker detection not implemented"
        },
        {
            "query": "Who declared 'You shall not pass'?",
            "expected_answer": "Gandalf",
            "context": "Gandalf confronting the Balrog in the Mines of Moria",
            "failure_reason": "No speaker identification capability"
        }
    ]
    
    evaluation_points = []
    
    for test in speaker_tests:
        evaluation_points.append(EvaluationDataPoint(
            query=test["query"],
            expected_answer=test["expected_answer"],
            relevant_document_ids=["placeholder_doc_id"],
            ground_truth_contexts=[test["context"]],
            evaluation_type="missing_feature",
            metadata={
                "feature_name": "speaker_attribution",
                "expected_to_fail": True,
                "failure_reason": test["failure_reason"],
                "missing_since": "2024-01-01",  # Track how long feature has been missing
                "priority": "high",
                "estimated_effort": "medium"
            }
        ))
    
    logger.info(
        "Created speaker attribution tests",
        num_tests=len(evaluation_points),
        expected_failures=len(evaluation_points),
        component="missing_features_tests"
    )
    
    return evaluation_points


def create_hybrid_search_tests() -> List[EvaluationDataPoint]:
    """Create tests for BM25 + semantic search combination.
    
    These tests check whether the system can perform hybrid search combining
    keyword matching (BM25) with semantic similarity - not yet implemented.
    
    Returns:
        List of hybrid search test cases that should fail
    """
    logger = get_logger(__name__)
    
    hybrid_search_tests = [
        {
            "query": "Find exact phrase 'ring' with semantic context about power",
            "expected_behavior": "Should find exact word 'ring' AND semantically related content about power",
            "current_limitation": "Only semantic search, no keyword matching",
            "failure_reason": "BM25 + semantic hybrid search not implemented"
        },
        {
            "query": "Search for 'Gandalf' AND related magic concepts",
            "expected_behavior": "Should combine exact name matching with semantic magic/wizard concepts",
            "current_limitation": "Purely vector-based search",
            "failure_reason": "No BM25 keyword component in retrieval pipeline"
        }
    ]
    
    evaluation_points = []
    
    for test in hybrid_search_tests:
        evaluation_points.append(EvaluationDataPoint(
            query=test["query"],
            expected_answer=None,  # Hybrid search is about retrieval quality, not specific answers
            relevant_document_ids=["placeholder_doc_id"],
            ground_truth_contexts=[test["expected_behavior"]],
            evaluation_type="missing_feature",
            metadata={
                "feature_name": "hybrid_search",
                "expected_to_fail": True,
                "failure_reason": test["failure_reason"],
                "current_limitation": test["current_limitation"],
                "missing_since": "2024-01-01",
                "priority": "high",
                "estimated_effort": "high"
            }
        ))
    
    return evaluation_points


def create_conversation_threading_tests() -> List[EvaluationDataPoint]:
    """Create tests for dialogue context understanding.
    
    These tests check whether the system can understand conversation flow
    and group related dialogue - not yet implemented.
    
    Returns:
        List of conversation threading test cases that should fail
    """
    logger = get_logger(__name__)
    
    conversation_tests = [
        {
            "query": "What was the conversation between Frodo and Sam about the ring?",
            "expected_behavior": "Should group multiple related dialogue exchanges",
            "failure_reason": "No conversation threading or dialogue grouping"
        },
        {
            "query": "Show me the full discussion during the Council of Elrond",
            "expected_behavior": "Should identify and group all related dialogue from the scene",
            "failure_reason": "Cannot identify conversation boundaries or participants"
        }
    ]
    
    evaluation_points = []
    
    for test in conversation_tests:
        evaluation_points.append(EvaluationDataPoint(
            query=test["query"],
            expected_answer=None,
            relevant_document_ids=["placeholder_doc_id"],
            ground_truth_contexts=[test["expected_behavior"]],
            evaluation_type="missing_feature",
            metadata={
                "feature_name": "conversation_threading",
                "expected_to_fail": True,
                "failure_reason": test["failure_reason"],
                "missing_since": "2024-01-01",
                "priority": "medium",
                "estimated_effort": "high"
            }
        ))
    
    return evaluation_points


def create_content_type_detection_tests() -> List[EvaluationDataPoint]:
    """Create tests for dialogue vs narration distinction.
    
    Returns:
        List of content type detection test cases that should fail
    """
    logger = get_logger(__name__)
    
    content_type_tests = [
        {
            "query": "Find dialogue between characters, not narration",
            "expected_behavior": "Should distinguish dialogue from narrative description",
            "failure_reason": "No content type classification implemented"
        },
        {
            "query": "Show me only narrative descriptions, not character speech",
            "expected_behavior": "Should filter for narration vs dialogue content",
            "failure_reason": "Cannot classify content types"
        }
    ]
    
    evaluation_points = []
    
    for test in content_type_tests:
        evaluation_points.append(EvaluationDataPoint(
            query=test["query"],
            expected_answer=None,
            relevant_document_ids=["placeholder_doc_id"],
            ground_truth_contexts=[test["expected_behavior"]],
            evaluation_type="missing_feature",
            metadata={
                "feature_name": "content_type_detection",
                "expected_to_fail": True,
                "failure_reason": test["failure_reason"],
                "missing_since": "2024-01-01",
                "priority": "medium",
                "estimated_effort": "low"
            }
        ))
    
    return evaluation_points


def track_missing_feature_duration(feature_name: str) -> Dict[str, Any]:
    """Track how long a feature has been missing/requested.
    
    Args:
        feature_name: Name of the missing feature
        
    Returns:
        Dictionary with timeline information about the missing feature
    """
    logger = get_logger(__name__)
    
    # TODO: Implement persistent tracking (could use file-based or database storage)
    # For now, use hardcoded baseline dates
    
    missing_since_dates = {
        "speaker_attribution": "2024-01-01",
        "hybrid_search": "2024-01-01", 
        "conversation_threading": "2024-01-01",
        "content_type_detection": "2024-01-01"
    }
    
    if feature_name not in missing_since_dates:
        logger.warning(f"Unknown feature for tracking: {feature_name}")
        return {}
    
    missing_since = datetime.fromisoformat(missing_since_dates[feature_name])
    duration = datetime.now() - missing_since
    
    return {
        "feature_name": feature_name,
        "missing_since": missing_since.isoformat(),
        "days_missing": duration.days,
        "status": "not_implemented",
        "last_checked": datetime.now().isoformat()
    }


def create_missing_features_evaluator(llm_api_key: str = None) -> LLMEvaluator:
    """Create Haystack LLMEvaluator for missing feature detection.
    
    Args:
        llm_api_key: API key for LLM service (Groq, OpenAI, etc.)
        
    Returns:
        Configured LLMEvaluator that detects missing features
    """
    from typing import List
    
    missing_features_evaluator = LLMEvaluator(
        instructions=(
            "Evaluate if the RAG system response includes the following capabilities:\n"
            "1. Speaker attribution (who said what)\n"
            "2. Hybrid search (exact keyword + semantic matching)\n"
            "3. Conversation threading (grouping related dialogue)\n"
            "4. Content type detection (dialogue vs narration)\n\n"
            "For each capability, return PASS if present, FAIL if missing.\n"
            "Provide specific evidence for your evaluation.\n"
            "Score: 0.0 if all features missing, 1.0 if all present."
        ),
        inputs=[("responses", List[str])],
        outputs=["score"],
        examples=[
            {
                "inputs": {"responses": "Gandalf said the quote but no speaker info provided"},
                "outputs": {"score": 0}
            },
            {
                "inputs": {"responses": "Speaker: Gandalf - 'You shall not pass!'"},
                "outputs": {"score": 1}
            }
        ]
    )
    
    return missing_features_evaluator


def generate_feature_roadmap_data() -> List[Dict[str, Any]]:
    """Generate roadmap data for missing features visualization.
    
    Returns:
        List of feature roadmap entries for dashboard visualization
    """
    features = [
        {
            "name": "Speaker Attribution",
            "status": "not_started",
            "priority": "high",
            "estimated_effort_weeks": 2,
            "dependencies": ["audio_processing", "speaker_diarization"],
            "expected_completion": "2024-04-01"
        },
        {
            "name": "Hybrid Search (BM25 + Semantic)",
            "status": "not_started", 
            "priority": "high",
            "estimated_effort_weeks": 3,
            "dependencies": ["bm25_integration", "result_fusion"],
            "expected_completion": "2024-03-15"
        },
        {
            "name": "Conversation Threading",
            "status": "not_started",
            "priority": "medium",
            "estimated_effort_weeks": 4,
            "dependencies": ["speaker_attribution", "dialogue_detection"],
            "expected_completion": "2024-05-01"
        },
        {
            "name": "Content Type Detection",
            "status": "not_started",
            "priority": "medium", 
            "estimated_effort_weeks": 1,
            "dependencies": [],
            "expected_completion": "2024-02-15"
        }
    ]
    
    return features