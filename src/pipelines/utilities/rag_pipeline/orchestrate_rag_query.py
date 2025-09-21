



"""Orchestration function for RAG query processing.

This module handles the complete RAG query pipeline:
1. Validates query parameters
2. Executes retrieval and generation through Haystack pipeline
3. Processes and formats results for consumption
4. Provides comprehensive error handling and logging

Follows Memoirr patterns: pure functions, proper error handling, comprehensive logging.
"""

from typing import Dict, Any, Optional

from haystack import Pipeline

from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger


def orchestrate_rag_query(
    pipeline: Pipeline,
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Orchestrate a complete RAG query through the pipeline.

    Args:
        pipeline: Configured Haystack pipeline with retriever and generator.
        query: User question to answer.
        top_k: Override retrieval top_k.
        score_threshold: Override retrieval score threshold.
        filters: Metadata filters for retrieval.
        task_type: Task type for specialized generation.
        custom_instructions: Custom instructions for generation.
        max_tokens: Override generation max_tokens.
        temperature: Override generation temperature.

    Returns:
        Complete pipeline results with processed outputs.

    Raises:
        ValueError: If query is empty or invalid parameters.
        RuntimeError: If pipeline execution fails.
    """
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)

    # Validate input parameters
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only")

    with LoggedOperation("rag_orchestration", logger, query_length=len(query)) as op:
        try:
            # Step 1: Prepare pipeline inputs
            logger.debug(
                "Preparing pipeline inputs",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                top_k=top_k,
                score_threshold=score_threshold,
                has_filters=bool(filters),
                task_type=task_type,
                component="rag_orchestrator"
            )

            pipeline_inputs = _build_pipeline_inputs(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
                task_type=task_type,
                custom_instructions=custom_instructions,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Step 2: Execute the pipeline
            logger.debug(
                "Executing RAG pipeline",
                pipeline_components=list(pipeline.graph.nodes()),
                component="rag_orchestrator"
            )

            pipeline_result = pipeline.run(pipeline_inputs)

            logger.debug(
                "Pipeline execution completed",
                result_keys=list(pipeline_result.keys()),
                component="rag_orchestrator"
            )

            # Step 3: Process and validate results
            processed_result = _process_pipeline_result(pipeline_result, query)

            # Add operation context and metrics
            retriever_result = processed_result.get("retriever", {})
            generator_result = processed_result.get("generator", {})
            
            documents = retriever_result.get("documents", [])
            replies = generator_result.get("replies", [])
            meta = generator_result.get("meta", [])

            # Calculate metrics
            total_input_tokens = sum(m.get("usage", {}).get("prompt_tokens", 0) for m in meta)
            total_output_tokens = sum(m.get("usage", {}).get("completion_tokens", 0) for m in meta)

            op.add_context(
                retrieved_documents=len(documents),
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                pipeline_components=len(pipeline.graph.nodes())
            )

            metrics.counter("pipeline_executions_total", 1, component="rag_orchestrator", status="success")
            metrics.counter("pipeline_documents_retrieved", len(documents), component="rag_orchestrator")
            metrics.counter("pipeline_replies_generated", len(replies), component="rag_orchestrator")

            if documents:
                avg_score = sum(doc.score for doc in documents if doc.score) / len(documents)
                metrics.histogram("pipeline_avg_retrieval_score", avg_score, component="rag_orchestrator")

            logger.info(
                "RAG orchestration completed successfully",
                query_length=len(query),
                retrieved_documents=len(documents),
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                component="rag_orchestrator"
            )

            return processed_result

        except Exception as e:
            logger.error(
                "RAG orchestration failed",
                error=str(e),
                error_type=type(e).__name__,
                query_length=len(query),
                task_type=task_type,
                component="rag_orchestrator"
            )
            
            metrics.counter("pipeline_executions_total", 1, component="rag_orchestrator", status="error")
            metrics.counter("pipeline_errors_total", 1, component="rag_orchestrator", error_type=type(e).__name__)
            
            raise RuntimeError(f"RAG query failed: {str(e)}") from e


def _build_pipeline_inputs(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build pipeline inputs for both retriever and generator components.

    Args:
        query: User question.
        top_k: Retrieval parameter.
        score_threshold: Retrieval parameter.
        filters: Retrieval filters.
        task_type: Generation parameter.
        custom_instructions: Generation parameter.
        max_tokens: Generation parameter.
        temperature: Generation parameter.

    Returns:
        Dictionary of component inputs for pipeline execution.
    """
    # Build retriever inputs
    retriever_inputs = {"query": query}
    
    if top_k is not None:
        retriever_inputs["top_k"] = top_k
    if filters is not None:
        retriever_inputs["filters"] = filters

    # Build generator inputs
    generator_inputs = {"query": query}
    
    if task_type is not None:
        generator_inputs["task_type"] = task_type
    if custom_instructions is not None:
        generator_inputs["custom_instructions"] = custom_instructions
    if max_tokens is not None:
        generator_inputs["max_tokens"] = max_tokens
    if temperature is not None:
        generator_inputs["temperature"] = temperature

    return {
        "retriever": retriever_inputs,
        "generator": generator_inputs
    }


def _process_pipeline_result(pipeline_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """Process and validate pipeline results.

    Args:
        pipeline_result: Raw results from pipeline execution.
        original_query: Original user query for context.

    Returns:
        Processed and validated results.

    Raises:
        RuntimeError: If results are invalid or missing.
    """
    logger = get_logger(__name__)

    try:
        # Handle Haystack pipeline result structure - generator results contain both outputs
        if "generator" not in pipeline_result:
            raise RuntimeError("Missing generator results from pipeline")

        generator_result = pipeline_result["generator"]
        
        # Extract retriever info from generator output (which now includes documents)
        if "retriever" in pipeline_result:
            # Both outputs available (rare but possible)
            retriever_result = pipeline_result["retriever"]
        else:
            # Extract retriever info from generator output (common case)
            documents = generator_result.get("documents", [])
            retriever_result = {"documents": documents}

        # Validate retriever results
        if "documents" not in retriever_result:
            raise RuntimeError("Missing documents from retriever results")
        
        documents = retriever_result["documents"]
        if not isinstance(documents, list):
            raise RuntimeError("Documents should be a list")

        # Validate generator results
        if "replies" not in generator_result:
            raise RuntimeError("Missing replies from generator results")
        
        replies = generator_result["replies"]
        if not isinstance(replies, list):
            raise RuntimeError("Replies should be a list")

        # Add summary information
        processed_result = {
            "retriever": retriever_result,
            "generator": generator_result,
            "summary": {
                "query": original_query,
                "documents_retrieved": len(documents),
                "replies_generated": len(replies),
                "has_results": len(replies) > 0,
                "best_document_score": max((doc.score for doc in documents if doc.score), default=0.0),
            }
        }

        # Add answer extraction for convenience
        if replies:
            processed_result["answer"] = replies[0]  # Primary answer
            if len(replies) > 1:
                processed_result["alternative_answers"] = replies[1:]

        # Add source information
        if documents:
            processed_result["sources"] = [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "metadata": doc.meta or {}
                }
                for doc in documents
            ]

        logger.debug(
            "Pipeline results processed successfully",
            documents_retrieved=len(documents),
            replies_generated=len(replies),
            has_answer=bool(replies),
            component="rag_orchestrator"
        )

        return processed_result

    except Exception as e:
        logger.error(
            "Failed to process pipeline results",
            error=str(e),
            error_type=type(e).__name__,
            pipeline_result_keys=list(pipeline_result.keys()),
            component="rag_orchestrator"
        )
        raise RuntimeError(f"Failed to process pipeline results: {str(e)}") from e


def validate_rag_pipeline(pipeline: Pipeline) -> bool:
    """Validate that a pipeline is properly configured for RAG operations.

    Args:
        pipeline: Pipeline to validate.

    Returns:
        True if pipeline is valid for RAG operations.

    Raises:
        ValueError: If pipeline is invalid.
    """
    logger = get_logger(__name__)

    try:
        # Check that required components exist
        nodes = list(pipeline.graph.nodes())
        
        if "retriever" not in nodes:
            raise ValueError("Pipeline missing 'retriever' component")
        
        if "generator" not in nodes:
            raise ValueError("Pipeline missing 'generator' component")

        # Check that components are connected
        edges = list(pipeline.graph.edges())
        
        # Look for retriever -> generator connection
        retriever_to_generator = any(
            edge[0] == "retriever" and edge[1] == "generator"
            for edge in edges
        )
        
        if not retriever_to_generator:
            raise ValueError("Pipeline components not properly connected (retriever -> generator)")

        logger.debug(
            "Pipeline validation successful",
            components=nodes,
            connections=edges,
            component="rag_orchestrator"
        )

        return True

    except Exception as e:
        logger.error(
            "Pipeline validation failed",
            error=str(e),
            error_type=type(e).__name__,
            component="rag_orchestrator"
        )
        raise