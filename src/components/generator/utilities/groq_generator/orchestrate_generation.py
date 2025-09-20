"""Orchestration function for Groq text generation.

This module handles the complete generation pipeline:
1. Constructs appropriate prompts from documents and queries
2. Formats messages for Groq's OpenAI-compatible chat API
3. Calls Groq API with proper error handling and retry logic
4. Processes responses and returns formatted results

Follows Memoirr patterns: pure functions, proper error handling, comprehensive logging.
"""

from typing import List, Dict, Any, Optional, Tuple

from haystack.dataclasses import Document
from groq import Groq

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.generator.utilities.groq_generator.prompt_builder import build_rag_prompt
from src.components.generator.utilities.groq_generator.response_processor import process_groq_response


def orchestrate_generation(
    query: str,
    documents: List[Document],
    model: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stream: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Orchestrate the complete text generation process using Groq API.

    Args:
        query: The user question or prompt.
        documents: List of retrieved documents to use as context.
        model: Groq model name (e.g., "llama3-8b-8192").
        system_prompt: Optional system message to set behavior.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter (0.0-1.0).
        stream: Whether to stream responses (not yet implemented).

    Returns:
        Tuple of:
        - List of generated response strings
        - List of metadata dictionaries for each response

    Raises:
        ValueError: If query is empty or invalid parameters.
        RuntimeError: If Groq API call fails.
    """
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)
    settings = get_settings()

    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only")
    
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    if not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    if not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")

    with LoggedOperation("generation_orchestration", logger, query_length=len(query)) as op:
        try:
            # Step 1: Build the RAG prompt from documents and query
            logger.debug(
                "Building RAG prompt",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                document_count=len(documents),
                component="generation_orchestrator"
            )

            rag_prompt = build_rag_prompt(query, documents)
            
            logger.debug(
                "RAG prompt built successfully",
                prompt_length=len(rag_prompt),
                component="generation_orchestrator"
            )

            # Step 2: Construct messages for Groq chat API
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": rag_prompt
            })

            logger.debug(
                "Chat messages constructed",
                message_count=len(messages),
                has_system_prompt=bool(system_prompt),
                component="generation_orchestrator"
            )

            # Step 3: Initialize Groq client and make API call
            logger.debug(
                "Initializing Groq client",
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                component="generation_orchestrator"
            )

            client = Groq(api_key=settings.groq_api_key)

            # Prepare API parameters
            api_params = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            # Make the API call
            logger.debug(
                "Calling Groq API",
                model=model,
                input_messages=len(messages),
                component="generation_orchestrator"
            )

            response = client.chat.completions.create(**api_params)

            logger.debug(
                "Groq API call completed",
                response_id=getattr(response, 'id', 'unknown'),
                choice_count=len(response.choices) if response.choices else 0,
                component="generation_orchestrator"
            )

            # Step 4: Process the response
            replies, meta = process_groq_response(
                response=response,
                model=model,
                query=query,
                documents=documents,
                generation_params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

            # Add operation context and metrics
            total_input_tokens = sum(m.get("usage", {}).get("prompt_tokens", 0) for m in meta)
            total_output_tokens = sum(m.get("usage", {}).get("completion_tokens", 0) for m in meta)

            op.add_context(
                prompt_length=len(rag_prompt),
                message_count=len(messages),
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                model_used=model
            )

            metrics.counter("groq_api_calls_total", 1, component="generation_orchestrator", model=model, status="success")
            metrics.counter("prompts_built_total", 1, component="generation_orchestrator")
            metrics.counter("messages_sent_total", len(messages), component="generation_orchestrator")

            if total_input_tokens > 0:
                metrics.histogram("input_token_count", total_input_tokens, component="generation_orchestrator", model=model)
            if total_output_tokens > 0:
                metrics.histogram("output_token_count", total_output_tokens, component="generation_orchestrator", model=model)

            logger.info(
                "Generation orchestration completed successfully",
                query_length=len(query),
                document_count=len(documents),
                prompt_length=len(rag_prompt),
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                model=model,
                component="generation_orchestrator"
            )

            return replies, meta

        except Exception as e:
            logger.error(
                "Generation orchestration failed",
                error=str(e),
                error_type=type(e).__name__,
                query_length=len(query),
                document_count=len(documents),
                model=model,
                component="generation_orchestrator"
            )
            
            metrics.counter("groq_api_calls_total", 1, component="generation_orchestrator", model=model, status="error")
            metrics.counter("generation_errors_total", 1, component="generation_orchestrator", error_type=type(e).__name__)
            
            raise RuntimeError(f"Text generation failed: {str(e)}") from e