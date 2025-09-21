"""RAG (Retrieval-Augmented Generation) pipeline for subtitle question answering.

This pipeline combines document retrieval and text generation to answer questions
about movie/TV subtitle content using semantic search and AI generation.

Pipeline flow: Query → QdrantRetriever → GroqGenerator → Answer

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Dict, Any, Optional

from haystack.core.pipeline import Pipeline

from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.retriever.qdrant_retriever import QdrantRetriever
from src.components.generator.groq_generator import GroqGenerator
from src.pipelines.utilities.rag_pipeline.orchestrate_rag_query import orchestrate_rag_query


def build_rag_pipeline(
    retriever_config: Optional[Dict[str, Any]] = None,
    generator_config: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """Build a complete RAG pipeline for subtitle question answering.

    Args:
        retriever_config: Optional configuration overrides for QdrantRetriever.
        generator_config: Optional configuration overrides for GroqGenerator.

    Returns:
        Configured Haystack Pipeline ready for queries.
    """
    logger = get_logger(__name__)
    
    logger.info(
        "Building RAG pipeline",
        has_retriever_config=bool(retriever_config),
        has_generator_config=bool(generator_config),
        component="rag_pipeline_builder"
    )

    # Initialize components with optional overrides
    retriever_params = retriever_config or {}
    generator_params = generator_config or {}
    
    retriever = QdrantRetriever(**retriever_params)
    generator = GroqGenerator(**generator_params)

    # Build the pipeline
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("generator", generator)

    # Connect components: retriever documents → generator context
    pipeline.connect("retriever.documents", "generator.documents")

    logger.info(
        "RAG pipeline built successfully",
        components=["retriever", "generator"],
        connections=[("retriever.documents", "generator.documents")],
        component="rag_pipeline_builder"
    )

    return pipeline


def run_rag_query(
    pipeline: Pipeline,
    query: str,
    *,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a RAG query through the pipeline with optional parameter overrides.

    Args:
        pipeline: Configured RAG pipeline.
        query: User question to answer.
        top_k: Override retrieval top_k.
        score_threshold: Override retrieval score threshold.
        filters: Metadata filters for retrieval.
        task_type: Task type for specialized generation.
        custom_instructions: Custom instructions for generation.
        max_tokens: Override generation max_tokens.
        temperature: Override generation temperature.

    Returns:
        Dict containing the pipeline results with processed outputs.
    """
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)

    with LoggedOperation("rag_query", logger, query_length=len(query)) as op:
        logger.info(
            "Starting RAG query",
            query_preview=query[:100] + "..." if len(query) > 100 else query,
            top_k=top_k,
            score_threshold=score_threshold,
            has_filters=bool(filters),
            task_type=task_type,
            component="rag_pipeline"
        )

        # Delegate to orchestration function
        result = orchestrate_rag_query(
            pipeline=pipeline,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            task_type=task_type,
            custom_instructions=custom_instructions,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract metrics from result
        retriever_result = result.get("retriever", {})
        generator_result = result.get("generator", {})
        
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
            task_type=task_type
        )

        metrics.counter("rag_queries_total", 1, component="rag_pipeline", task_type=task_type or "general")
        metrics.counter("documents_retrieved_for_rag", len(documents), component="rag_pipeline")
        metrics.counter("replies_generated_for_rag", len(replies), component="rag_pipeline")

        if total_input_tokens > 0:
            metrics.histogram("rag_input_tokens", total_input_tokens, component="rag_pipeline")
        if total_output_tokens > 0:
            metrics.histogram("rag_output_tokens", total_output_tokens, component="rag_pipeline")

        logger.info(
            "RAG query completed",
            retrieved_documents=len(documents),
            generated_replies=len(replies),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            component="rag_pipeline"
        )

        return result


class RAGPipeline:
    """High-level interface for subtitle RAG operations.
    
    Provides a convenient wrapper around the Haystack pipeline for common
    subtitle question-answering tasks with built-in optimizations.
    """

    def __init__(
        self,
        retriever_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize RAG pipeline with optional component configurations.
        
        Args:
            retriever_config: Configuration overrides for QdrantRetriever.
            generator_config: Configuration overrides for GroqGenerator.
        """
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        self._pipeline = build_rag_pipeline(
            retriever_config=retriever_config,
            generator_config=generator_config
        )
        
        self._logger.info(
            "RAGPipeline initialized",
            component="rag_pipeline_wrapper"
        )

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: User question to answer.
            top_k: Maximum documents to retrieve.
            score_threshold: Minimum similarity score for retrieved documents.
            filters: Metadata filters (e.g., {"speaker": "Tony Stark"}).
            task_type: Specialized task type ("character_analysis", "quote_finding", etc.).
            custom_instructions: Additional instructions for the AI.
            max_tokens: Maximum tokens to generate.
            temperature: Generation creativity (0.0-2.0).

        Returns:
            Dict with answer, sources, metadata, and usage information.
        """
        return run_rag_query(
            pipeline=self._pipeline,
            query=question,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            task_type=task_type,
            custom_instructions=custom_instructions,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def character_analysis(
        self,
        question: str,
        character_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Specialized method for character analysis questions.

        Args:
            question: Question about character development, relationships, etc.
            character_name: Optional character to focus on.
            **kwargs: Additional parameters passed to query().

        Returns:
            Character analysis response with context.
        """
        filters = kwargs.pop("filters", {})
        if character_name:
            filters["speaker"] = character_name

        return self.query(
            question=question,
            task_type="character_analysis",
            filters=filters,
            **kwargs
        )

    def find_quote(
        self,
        quote_text: str,
        speaker: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Specialized method for finding specific quotes or dialogue.

        Args:
            quote_text: Quote or dialogue to find.
            speaker: Optional speaker to filter by.
            **kwargs: Additional parameters passed to query().

        Returns:
            Quote finding response with exact matches and timestamps.
        """
        filters = kwargs.pop("filters", {})
        if speaker:
            filters["speaker"] = speaker

        # Use higher similarity threshold for quote finding
        score_threshold = kwargs.pop("score_threshold", 0.8)

        return self.query(
            question=f"Find this exact quote or similar dialogue: {quote_text}",
            task_type="quote_finding",
            filters=filters,
            score_threshold=score_threshold,
            **kwargs
        )

    def timeline_analysis(
        self,
        question: str,
        time_range: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Specialized method for timeline and chronological questions.

        Args:
            question: Question about timing, sequence, or chronology.
            time_range: Optional time range filter {"start_seconds": x, "end_seconds": y}.
            **kwargs: Additional parameters passed to query().

        Returns:
            Timeline analysis response with temporal context.
        """
        filters = kwargs.pop("filters", {})
        if time_range:
            if "start_seconds" in time_range:
                filters["start_ms"] = {"gte": time_range["start_seconds"] * 1000}
            if "end_seconds" in time_range:
                filters["end_ms"] = {"lte": time_range["end_seconds"] * 1000}

        return self.query(
            question=question,
            task_type="timeline",
            filters=filters,
            **kwargs
        )