"""Haystack component for text generation using Groq's OpenAI-compatible API.

This component leverages Groq's fast inference for chat completion tasks in RAG pipelines.
It uses the Groq Python client which provides OpenAI-compatible endpoints and interfaces.

It follows Haystack's custom component requirements:
- @component decorator  
- run() method returning a dict matching @component.output_types

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import List, Dict, Any, Optional

from haystack import component
from haystack.dataclasses import Document

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.components.generator.utilities.groq_generator.orchestrate_generation import orchestrate_generation
from src.prompts.template_loader import render_system_prompt


@component
class GroqGenerator:
    """Haystack component for generating text responses using Groq's API.

    This component handles the complete generation pipeline:
    1. Constructs prompts from documents and queries
    2. Calls Groq's chat completion API 
    3. Returns generated responses with metadata

    Args:
        model: Groq model name (e.g., "llama3-8b-8192", "mixtral-8x7b-32768").
        system_prompt: Optional system message to set context/behavior.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter (0.0-1.0). 
        stream: Whether to stream responses (currently not implemented).
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: Optional[bool] = None,
    ) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)

        # Resolve configuration from args or .env
        self.model = model if model is not None else settings.groq_model
        self.system_prompt_template = system_prompt_template if system_prompt_template is not None else settings.groq_system_prompt_template
        self.max_tokens = max_tokens if max_tokens is not None else settings.groq_max_tokens
        self.temperature = temperature if temperature is not None else settings.groq_temperature
        self.top_p = top_p if top_p is not None else settings.groq_top_p
        self.stream = stream if stream is not None else settings.groq_stream

        self._logger.info(
            "GroqGenerator initialized successfully",
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.stream,
            system_prompt_template=self.system_prompt_template,
            component="groq_generator"
        )

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self, 
        query: str, 
        documents: Optional[List[Document]] = None,
        task_type: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, object]:  # type: ignore[override]
        """Generate a response using Groq's chat completion API.

        Args:
            query: The user question or prompt.
            documents: Optional list of retrieved documents to use as context.
            task_type: Optional task type for specialized system prompt (e.g., "character_analysis").
            custom_instructions: Optional custom instructions to add to system prompt.
            system_prompt_template: Override the default template for this request.
            max_tokens: Override the default max tokens for this request.
            temperature: Override the default temperature for this request.
            top_p: Override the default top_p for this request.

        Returns:
            Dict with:
            - replies: List of generated response strings
            - meta: List of metadata for each response (model info, token usage, etc.)
        """
        with LoggedOperation("groq_generation", self._logger, query_length=len(query)) as op:
            # Use provided parameters or fall back to instance defaults
            effective_template = system_prompt_template if system_prompt_template is not None else self.system_prompt_template
            effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_top_p = top_p if top_p is not None else self.top_p

            # Render system prompt from template
            try:
                rendered_system_prompt = render_system_prompt(
                    template_name=effective_template,
                    task_type=task_type,
                    custom_instructions=custom_instructions
                )
            except Exception as e:
                self._logger.warning(
                    "Failed to render system prompt template, using fallback",
                    template_name=effective_template,
                    error=str(e),
                    component="groq_generator"
                )
                # Fallback to a basic system prompt
                rendered_system_prompt = "You are a helpful assistant that answers questions based on provided context."

            self._logger.info(
                "Starting text generation",
                query_preview=query[:100] + "..." if len(query) > 100 else query,
                document_count=len(documents) if documents else 0,
                model=self.model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                task_type=task_type,
                template_used=effective_template,
                component="groq_generator"
            )

            # Delegate to orchestration function
            replies, meta = orchestrate_generation(
                query=query,
                documents=documents or [],
                model=self.model,
                system_prompt=rendered_system_prompt,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                top_p=effective_top_p,
                stream=self.stream,
            )

            # Add context and metrics
            total_input_tokens = sum(m.get("usage", {}).get("prompt_tokens", 0) for m in meta)
            total_output_tokens = sum(m.get("usage", {}).get("completion_tokens", 0) for m in meta)
            
            op.add_context(
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                model_used=self.model,
                document_count=len(documents) if documents else 0
            )

            self._metrics.counter("generation_requests_total", 1, component="groq_generator", model=self.model)
            self._metrics.counter("replies_generated_total", len(replies), component="groq_generator", model=self.model)
            self._metrics.counter("tokens_generated_total", total_output_tokens, component="groq_generator", model=self.model, token_type="output")
            self._metrics.counter("tokens_consumed_total", total_input_tokens, component="groq_generator", model=self.model, token_type="input")

            if replies:
                avg_reply_length = sum(len(reply) for reply in replies) / len(replies)
                self._metrics.histogram("reply_length_chars", avg_reply_length, component="groq_generator", model=self.model)

            self._logger.info(
                "Text generation completed",
                generated_replies=len(replies),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                avg_reply_length=sum(len(reply) for reply in replies) / len(replies) if replies else 0,
                component="groq_generator"
            )

            return {"replies": replies, "meta": meta}