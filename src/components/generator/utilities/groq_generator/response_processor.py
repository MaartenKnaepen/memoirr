"""Utility functions for processing Groq API responses.

This module handles response parsing, metadata extraction, and formatting
of generated text for downstream consumption in RAG pipelines.

Follows Memoirr patterns: pure functions, comprehensive error handling, structured metadata.
"""

from typing import List, Dict, Any, Tuple, Optional

from haystack.dataclasses import Document

from src.core.logging_config import get_logger


def process_groq_response(
    response: Any,
    model: str,
    query: str,
    documents: List[Document],
    generation_params: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process a Groq chat completion response into replies and metadata.

    Args:
        response: The Groq ChatCompletion response object.
        model: Model name used for generation.
        query: Original user query.
        documents: Documents that were used as context.
        generation_params: Parameters used for generation (temperature, max_tokens, etc.).

    Returns:
        Tuple of:
        - List of generated reply strings
        - List of metadata dictionaries for each reply

    Raises:
        ValueError: If response format is invalid.
        RuntimeError: If response processing fails.
    """
    logger = get_logger(__name__)
    
    try:
        logger.debug(
            "Processing Groq response",
            response_id=getattr(response, 'id', 'unknown'),
            choice_count=len(response.choices) if response.choices else 0,
            component="response_processor"
        )

        # Extract replies from response choices
        replies = []
        meta_list = []

        if not response.choices:
            logger.warning(
                "No choices in Groq response",
                response_id=getattr(response, 'id', 'unknown'),
                component="response_processor"
            )
            return [], []

        for i, choice in enumerate(response.choices):
            try:
                # Extract the generated text
                reply_text = _extract_reply_text(choice)
                replies.append(reply_text)

                # Build metadata for this reply
                meta = _build_reply_metadata(
                    choice=choice,
                    choice_index=i,
                    response=response,
                    model=model,
                    query=query,
                    documents=documents,
                    generation_params=generation_params,
                )
                meta_list.append(meta)

                logger.debug(
                    "Choice processed successfully",
                    choice_index=i,
                    reply_length=len(reply_text),
                    finish_reason=getattr(choice, 'finish_reason', 'unknown'),
                    component="response_processor"
                )

            except Exception as e:
                logger.warning(
                    "Failed to process choice",
                    choice_index=i,
                    error=str(e),
                    error_type=type(e).__name__,
                    component="response_processor"
                )
                # Continue processing other choices
                continue

        logger.debug(
            "Groq response processed successfully",
            total_choices=len(response.choices),
            successful_replies=len(replies),
            component="response_processor"
        )

        return replies, meta_list

    except Exception as e:
        logger.error(
            "Failed to process Groq response",
            error=str(e),
            error_type=type(e).__name__,
            response_id=getattr(response, 'id', 'unknown'),
            component="response_processor"
        )
        raise RuntimeError(f"Response processing failed: {str(e)}") from e


def _extract_reply_text(choice: Any) -> str:
    """Extract the text content from a response choice.

    Args:
        choice: A choice object from the Groq response.

    Returns:
        The generated text content.

    Raises:
        ValueError: If choice format is invalid.
    """
    try:
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            content = choice.message.content
            if content is not None:
                return content.strip()
        
        raise ValueError("Invalid choice format: missing message.content")
        
    except AttributeError as e:
        raise ValueError(f"Invalid choice structure: {str(e)}") from e


def _build_reply_metadata(
    choice: Any,
    choice_index: int,
    response: Any,
    model: str,
    query: str,
    documents: List[Document],
    generation_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build comprehensive metadata for a generated reply.

    Args:
        choice: The response choice object.
        choice_index: Index of this choice in the response.
        response: The full Groq response object.
        model: Model name used for generation.
        query: Original user query.
        documents: Documents used as context.
        generation_params: Generation parameters.

    Returns:
        Metadata dictionary for this reply.
    """
    meta = {
        # Basic reply info
        "choice_index": choice_index,
        "model": model,
        "finish_reason": getattr(choice, 'finish_reason', None),
        
        # Generation parameters
        "generation_params": generation_params.copy(),
        
        # Context info
        "query": query,
        "document_count": len(documents),
        "context_sources": _extract_context_sources(documents),
        
        # Response metadata
        "response_id": getattr(response, 'id', None),
        "created": getattr(response, 'created', None),
        "object": getattr(response, 'object', None),
    }

    # Add usage statistics if available
    usage = getattr(response, 'usage', None)
    if usage:
        meta["usage"] = {
            "prompt_tokens": getattr(usage, 'prompt_tokens', None),
            "completion_tokens": getattr(usage, 'completion_tokens', None),
            "total_tokens": getattr(usage, 'total_tokens', None),
        }

    # Add choice-specific metadata
    if hasattr(choice, 'index'):
        meta["choice_index_from_api"] = choice.index
    
    if hasattr(choice, 'logprobs') and choice.logprobs:
        meta["has_logprobs"] = True
        # Note: Could add more detailed logprob processing here if needed

    return meta


def _extract_context_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """Extract source information from context documents.

    Args:
        documents: List of documents used as context.

    Returns:
        List of source metadata dictionaries.
    """
    sources = []
    
    for i, doc in enumerate(documents):
        source = {
            "index": i,
            "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
            "content_length": len(doc.content),
        }
        
        if doc.meta:
            # Add relevant metadata fields
            for key in ["start_ms", "end_ms", "speaker", "caption_index", "retrieval_score"]:
                if key in doc.meta:
                    source[key] = doc.meta[key]
            
            # Add score from document if available
            if doc.score is not None:
                source["similarity_score"] = doc.score
        
        sources.append(source)
    
    return sources


def format_reply_for_display(reply: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Format a reply and its metadata for user display.

    Args:
        reply: The generated reply text.
        meta: Reply metadata.

    Returns:
        Display-formatted dictionary with reply and relevant metadata.
    """
    display_info = {
        "answer": reply,
        "model": meta.get("model"),
        "sources_used": len(meta.get("context_sources", [])),
    }

    # Add token usage if available
    usage = meta.get("usage", {})
    if usage.get("total_tokens"):
        display_info["token_usage"] = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        }

    # Add generation quality indicators
    finish_reason = meta.get("finish_reason")
    if finish_reason:
        display_info["completion_status"] = finish_reason
        if finish_reason == "length":
            display_info["warning"] = "Response may be truncated due to length limits"

    # Add context summary
    sources = meta.get("context_sources", [])
    if sources:
        display_info["context_summary"] = {
            "document_count": len(sources),
            "time_range": _extract_time_range(sources),
            "speakers": _extract_speakers(sources),
        }

    return display_info


def _extract_time_range(sources: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Extract time range from context sources.

    Args:
        sources: List of source metadata.

    Returns:
        Time range dictionary or None if no timing info.
    """
    start_times = []
    end_times = []
    
    for source in sources:
        if "start_ms" in source and source["start_ms"] is not None:
            start_times.append(source["start_ms"] / 1000)  # Convert to seconds
        if "end_ms" in source and source["end_ms"] is not None:
            end_times.append(source["end_ms"] / 1000)
    
    if start_times and end_times:
        return {
            "start_seconds": min(start_times),
            "end_seconds": max(end_times),
            "duration_seconds": max(end_times) - min(start_times),
        }
    
    return None


def _extract_speakers(sources: List[Dict[str, Any]]) -> List[str]:
    """Extract unique speakers from context sources.

    Args:
        sources: List of source metadata.

    Returns:
        List of unique speaker names.
    """
    speakers = set()
    
    for source in sources:
        speaker = source.get("speaker")
        if speaker:
            speakers.add(speaker)
    
    return sorted(list(speakers))