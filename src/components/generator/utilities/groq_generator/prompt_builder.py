"""Utility functions for building RAG prompts from retrieved documents.

This module handles prompt construction for subtitle-based RAG queries,
including context formatting, document ranking, and query integration.

Follows Memoirr patterns: pure functions, clear interfaces, subtitle-specific optimizations.
"""

from typing import List

from haystack.dataclasses import Document

from src.core.config import get_settings
from src.core.logging_config import get_logger


def build_rag_prompt(query: str, documents: List[Document]) -> str:
    """Build a RAG prompt from the user query and retrieved documents.

    Constructs a prompt that includes relevant subtitle context while
    maintaining readability and staying within token limits.

    Args:
        query: The user's question or request.
        documents: List of retrieved subtitle documents with metadata.

    Returns:
        Formatted prompt string ready for the language model.
    """
    settings = get_settings()
    logger = get_logger(__name__)
    
    if not documents:
        logger.debug(
            "No documents provided, using query-only prompt",
            query_length=len(query),
            component="prompt_builder"
        )
        return _build_query_only_prompt(query)
    
    logger.debug(
        "Building RAG prompt with context",
        query_length=len(query),
        document_count=len(documents),
        component="prompt_builder"
    )
    
    # Format the context from documents
    context_sections = []
    total_context_length = 0
    max_context_length = settings.groq_max_context_length
    
    for i, doc in enumerate(documents):
        section = _format_document_context(doc, i + 1)
        
        # Check if adding this section would exceed context limit
        if total_context_length + len(section) > max_context_length:
            logger.debug(
                "Context limit reached, truncating documents",
                included_documents=i,
                total_documents=len(documents),
                context_length=total_context_length,
                max_context_length=max_context_length,
                component="prompt_builder"
            )
            break
            
        context_sections.append(section)
        total_context_length += len(section)
    
    context = "\n".join(context_sections)
    
    # Build the complete prompt
    prompt = _build_contextualized_prompt(query, context)
    
    logger.debug(
        "RAG prompt built successfully",
        final_prompt_length=len(prompt),
        context_length=len(context),
        included_documents=len(context_sections),
        component="prompt_builder"
    )
    
    return prompt


def _build_query_only_prompt(query: str) -> str:
    """Build a prompt for queries without retrieved context.
    
    Args:
        query: The user's question.
        
    Returns:
        Simple prompt string.
    """
    return f"""Please answer the following question:

Question: {query}

Answer:"""


def _build_contextualized_prompt(query: str, context: str) -> str:
    """Build a prompt that includes retrieved subtitle context.
    
    Args:
        query: The user's question.
        context: Formatted context from retrieved documents.
        
    Returns:
        Complete RAG prompt string.
    """
    return f"""You are a helpful assistant that answers questions based on provided subtitle/transcript context. Use the following context to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context from subtitles/transcripts:
{context}

Question: {query}

Answer:"""


def _format_document_context(document: Document, rank: int) -> str:
    """Format a single document as context for the prompt.
    
    Args:
        document: Retrieved document with content and metadata.
        rank: Document ranking (1-based).
        
    Returns:
        Formatted context string for this document.
    """
    content = document.content.strip()
    
    # Build metadata info
    meta_parts = []
    
    if document.meta:
        # Add timestamp information if available
        start_ms = document.meta.get("start_ms")
        end_ms = document.meta.get("end_ms")
        if start_ms is not None and end_ms is not None:
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            meta_parts.append(f"Time: {start_sec:.1f}s - {end_sec:.1f}s")
        
        # Add speaker information if available
        speaker = document.meta.get("speaker")
        if speaker:
            meta_parts.append(f"Speaker: {speaker}")
        
        # Add retrieval score if available
        score = document.meta.get("retrieval_score") or document.score
        if score is not None:
            meta_parts.append(f"Relevance: {score:.3f}")
    
    # Format the complete section
    if meta_parts:
        meta_info = " | ".join(meta_parts)
        return f"[{rank}] ({meta_info})\n{content}"
    else:
        return f"[{rank}] {content}"


def build_system_prompt(task_type: str = "general") -> str:
    """Build system prompts for different types of subtitle RAG tasks.
    
    Args:
        task_type: Type of task ("general", "character_analysis", "quote_finding", "timeline").
        
    Returns:
        Appropriate system prompt for the task.
    """
    prompts = {
        "general": """You are a helpful assistant that answers questions about movies, TV shows, and other video content based on subtitle/transcript information. Provide accurate, helpful responses based on the provided context.""",
        
        "character_analysis": """You are a character analysis expert. Use the provided subtitle/transcript context to analyze character development, relationships, and dialogue patterns. Focus on what characters say and how they interact.""",
        
        "quote_finding": """You are a quote finder. Help users locate specific dialogue or memorable lines from the provided subtitle/transcript context. Be precise about attribution and timing when possible.""",
        
        "timeline": """You are a timeline expert. Use the provided subtitle/transcript context to answer questions about when events happen, the sequence of scenes, and temporal relationships in the content.""",
    }
    
    return prompts.get(task_type, prompts["general"])


def truncate_context_to_limit(context: str, max_length: int) -> str:
    """Truncate context to fit within token/character limits.
    
    Args:
        context: The full context string.
        max_length: Maximum allowed length in characters.
        
    Returns:
        Truncated context that fits within the limit.
    """
    if len(context) <= max_length:
        return context
    
    # Truncate and add indicator
    truncated = context[:max_length - 50]  # Leave room for truncation message
    
    # Try to break at a natural boundary (end of a document section)
    last_bracket = truncated.rfind("[")
    if last_bracket > max_length * 0.8:  # Only if we don't lose too much
        truncated = truncated[:last_bracket]
    
    return truncated + "\n\n[... context truncated due to length limits ...]"