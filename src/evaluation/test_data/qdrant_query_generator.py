"""Generate evaluation queries from existing Qdrant collections.

This module connects to Qdrant collections containing processed subtitle data
and generates realistic test queries for RAG evaluation.
"""

from typing import List, Dict, Any, Optional
import random

from src.core.logging_config import get_logger
from src.evaluation.test_data.ground_truth_builder import EvaluationDataPoint


def generate_queries_from_qdrant(
    collection_name: str, 
    num_queries: int = 30,
    query_distribution: Optional[Dict[str, float]] = None
) -> List[EvaluationDataPoint]:
    """Generate test queries directly from existing Qdrant LOTR collection.
    
    Args:
        collection_name: Name of Qdrant collection with processed subtitles
        num_queries: Total number of queries to generate
        query_distribution: Distribution of query types (e.g., {"quote": 0.4, "plot": 0.6})
        
    Returns:
        List of evaluation data points generated from actual Qdrant data
        
    Example:
        queries = generate_queries_from_qdrant("lotr_evaluation", num_queries=25)
        quote_queries = [q for q in queries if q.evaluation_type == "exact_match"]
    """
    logger = get_logger(__name__)
    
    if query_distribution is None:
        query_distribution = {
            "exact_quote": 0.4,
            "plot_summary": 0.3,
            "temporal": 0.3
        }
    
    logger.info(
        "Generating queries from Qdrant collection",
        collection=collection_name,
        num_queries=num_queries,
        distribution=query_distribution,
        component="qdrant_query_generator"
    )
    
    try:
        # Connect to Qdrant and sample chunks
        qdrant_client = connect_to_qdrant_collection(collection_name)
        sampled_chunks = sample_chunks_for_evaluation(qdrant_client, num_samples=100)
        
        generated_queries = []
        
        # Generate queries based on distribution
        for query_type, ratio in query_distribution.items():
            num_for_type = int(num_queries * ratio)
            
            if query_type == "exact_quote":
                quote_queries = generate_exact_quote_queries(sampled_chunks[:num_for_type])
                generated_queries.extend(quote_queries)
            elif query_type == "plot_summary":
                plot_queries = generate_plot_summary_queries(sampled_chunks[:num_for_type])
                generated_queries.extend(plot_queries)
            elif query_type == "temporal":
                temporal_queries = generate_temporal_queries(sampled_chunks[:num_for_type])
                generated_queries.extend(temporal_queries)
        
        # Validate generated queries
        validated_queries = [q for q in generated_queries if validate_generated_queries([q])]
        
        logger.info(
            "Query generation completed",
            total_generated=len(generated_queries),
            validated=len(validated_queries),
            component="qdrant_query_generator"
        )
        
        return validated_queries[:num_queries]  # Ensure we don't exceed requested count
        
    except Exception as e:
        logger.error(
            "Query generation failed",
            error=str(e),
            error_type=type(e).__name__,
            component="qdrant_query_generator"
        )
        raise


def connect_to_qdrant_collection(collection_name: str):
    """Connect to Qdrant collection with existing LOTR data.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        QdrantClient instance configured for the collection
    """
    logger = get_logger(__name__)
    
    # TODO: Implement actual Qdrant connection
    # This will use the same connection logic as existing retriever components
    
    logger.info(
        "Connecting to Qdrant collection",
        collection=collection_name,
        component="qdrant_query_generator"
    )
    
    # Placeholder - will be implemented in Day 3
    return None


def sample_chunks_for_evaluation(client, num_samples: int = 100) -> List[Dict]:
    """Sample chunks from Qdrant collection for query generation.
    
    Args:
        client: QdrantClient instance
        num_samples: Number of chunks to sample
        
    Returns:
        List of sampled chunk documents with text and metadata
    """
    logger = get_logger(__name__)
    
    # TODO: Implement actual chunk sampling from Qdrant
    # This should randomly sample documents from the collection
    
    # Placeholder sample data based on LOTR content structure
    sample_chunks = [
        {
            "text": "I will take the Ring to Mordor, though I do not know the way.",
            "start_ms": 1234567,
            "end_ms": 1238567,
            "doc_id": "lotr_fellowship_chunk_089",
            "token_count": 15
        },
        {
            "text": "You cannot pass! I am a servant of the Secret Fire, wielder of the flame of Anor.",
            "start_ms": 2345678,
            "end_ms": 2349678,
            "doc_id": "lotr_fellowship_chunk_156",
            "token_count": 18
        }
    ]
    
    logger.info(
        "Sampled chunks for evaluation",
        num_chunks=len(sample_chunks),
        component="qdrant_query_generator"
    )
    
    return sample_chunks[:num_samples]


def generate_exact_quote_queries(chunks: List[Dict]) -> List[EvaluationDataPoint]:
    """Generate exact quote queries from chunk data.
    
    Args:
        chunks: List of chunk documents from Qdrant
        
    Returns:
        List of exact match evaluation data points
    """
    logger = get_logger(__name__)
    
    quote_queries = []
    
    for chunk in chunks:
        # Look for dialogue or memorable quotes
        text = chunk.get("text", "")
        
        # Simple heuristic: quotes with dialogue markers or emotional language
        if any(marker in text.lower() for marker in ['"', "'", "!", "said", "declared", "shouted"]):
            query_text = f"Find this exact quote: {text}"
            
            quote_queries.append(EvaluationDataPoint(
                query=query_text,
                expected_answer=text,
                relevant_document_ids=[chunk.get("doc_id", "unknown")],
                ground_truth_contexts=[text],
                evaluation_type="exact_match",
                metadata={
                    "source": "qdrant_generation",
                    "chunk_start_ms": chunk.get("start_ms"),
                    "chunk_end_ms": chunk.get("end_ms"),
                    "generation_method": "dialogue_heuristic"
                }
            ))
    
    logger.info(
        "Generated exact quote queries",
        num_queries=len(quote_queries),
        component="qdrant_query_generator"
    )
    
    return quote_queries


def generate_plot_summary_queries(chunks: List[Dict]) -> List[EvaluationDataPoint]:
    """Generate plot/narrative summary queries from chunk data.
    
    Args:
        chunks: List of chunk documents from Qdrant
        
    Returns:
        List of plot summary evaluation data points
    """
    logger = get_logger(__name__)
    
    plot_queries = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        
        # Look for narrative or action descriptions
        if any(action in text.lower() for action in ["went", "walked", "arrived", "happened", "battle"]):
            # Create a question about the events described
            query_text = f"What happened in this scene: {text[:50]}...?"
            
            plot_queries.append(EvaluationDataPoint(
                query=query_text,
                expected_answer=text,  # The full chunk is the expected context
                relevant_document_ids=[chunk.get("doc_id", "unknown")],
                ground_truth_contexts=[text],
                evaluation_type="faithfulness",  # Plot summaries test faithfulness
                metadata={
                    "source": "qdrant_generation",
                    "chunk_start_ms": chunk.get("start_ms"),
                    "generation_method": "plot_heuristic"
                }
            ))
    
    return plot_queries


def generate_temporal_queries(chunks: List[Dict]) -> List[EvaluationDataPoint]:
    """Generate temporal/timeline queries from chunk data.
    
    Args:
        chunks: List of chunk documents from Qdrant
        
    Returns:
        List of temporal evaluation data points
    """
    logger = get_logger(__name__)
    
    temporal_queries = []
    
    # Sort chunks by timestamp for temporal relationships
    sorted_chunks = sorted(chunks, key=lambda x: x.get("start_ms", 0))
    
    for i, chunk in enumerate(sorted_chunks[:-1]):  # Skip last chunk
        current_text = chunk.get("text", "")
        next_chunk = sorted_chunks[i + 1]
        next_text = next_chunk.get("text", "")
        
        # Create temporal relationship queries
        query_text = f"What happened after: {current_text[:30]}...?"
        
        temporal_queries.append(EvaluationDataPoint(
            query=query_text,
            expected_answer=next_text,
            relevant_document_ids=[next_chunk.get("doc_id", "unknown")],
            ground_truth_contexts=[next_text],
            evaluation_type="context_relevance",  # Temporal queries test context relevance
            metadata={
                "source": "qdrant_generation",
                "temporal_relationship": "sequential",
                "previous_chunk_id": chunk.get("doc_id"),
                "generation_method": "temporal_sequence"
            }
        ))
    
    return temporal_queries


def validate_generated_queries(queries: List[EvaluationDataPoint]) -> bool:
    """Validate generated queries against quality criteria.
    
    Args:
        queries: List of generated evaluation data points
        
    Returns:
        True if queries meet quality standards
    """
    logger = get_logger(__name__)
    
    if not queries:
        return False
    
    for query in queries:
        # Basic validation criteria
        if not query.query or len(query.query) < 10:
            logger.warning("Query too short", query=query.query)
            return False
            
        if not query.ground_truth_contexts:
            logger.warning("Missing ground truth context", query=query.query)
            return False
    
    return True