"""Haystack component to embed text using a local sentence-transformers model.

This component aligns with Memoirr's standards:
- Uses Haystack as the abstraction layer for RAG pipelines.
- Reads configuration from src.core.config.Settings (model name, device, dimension).
- Self-hosted, in-process model loaded from the local models/ folder.

It exposes a single input socket `text` and a single output `embedding`,
following the simple, JSON-serializable socket guidance in documentation/qwen.md.
"""

from typing import List

from haystack.core.component import component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from tqdm import tqdm

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.core.model_utils import resolve_model_path


@component
class TextEmbedder:
    """Embed text using a local sentence-transformers model.

    The model folder is resolved relative to the repository's `models/` directory using
    the `EMBEDDING_MODEL_NAME` environment variable (via Settings). This should match
    the model used by the semantic chunker to ensure consistent embedding spaces.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        # The SentenceTransformersTextEmbedder accepts a `model` argument that can be
        # a local folder path. We pass the Settings.embedding_model_name verbatim,
        # expecting users to place the model files in `models/<EMBEDDING_MODEL_NAME>`.
        # Device is managed internally by sentence-transformers; if needed, it can be
        # configured via environment or future extension.
        self._model_name = settings.embedding_model_name
        # Use explicit embedding dimension from settings, with fallback only as last resort
        if settings.embedding_dimension is not None:
            self._embedding_dimension = settings.embedding_dimension
        else:
            # Log warning about using fallback
            self._logger.warning(
                "Embedding dimension not configured, using fallback",
                fallback_dimension=settings.embedding_dimension_fallback,
                component="embedder",
                recommendation="Set EMBEDDING_DIMENSION in .env for proper configuration"
            )
            self._embedding_dimension = settings.embedding_dimension_fallback
        
        # Resolve model directory using common utility
        try:
            model_dir = resolve_model_path(self._model_name)
            self._logger.info(
                "Embedding model resolved successfully",
                model_name=self._model_name,
                model_path=str(model_dir),
                component="embedder"
            )
        except FileNotFoundError as e:
            self._logger.error(
                "Embedding model not found",
                model_name=self._model_name,
                error=str(e),
                component="embedder"
            )
            raise FileNotFoundError(f"Embedding model not found: {e}") from e
        # Initialize embedder with the resolved local path.
        # Prefer offline mode when supported by the installed Haystack version.
        # Disable progress bars to avoid individual 1/1 progress bars
        try:
            self._embedder = SentenceTransformersTextEmbedder(
                model=str(model_dir),
                local_files_only=True,
                show_progress_bar=False,
            )
        except TypeError:
            # Fallback for environments (or tests) where the underlying class signature
            # doesn't accept `local_files_only` or `show_progress_bar` (e.g., monkeypatched fakes).
            try:
                self._embedder = SentenceTransformersTextEmbedder(
                    model=str(model_dir),
                    show_progress_bar=False,
                )
            except TypeError:
                # Final fallback for minimal constructor
                self._embedder = SentenceTransformersTextEmbedder(model=str(model_dir))
        # Warm up so that the first embedding call in a pipeline is not cold.
        self._embedder.warm_up()
        
        self._logger.info(
            "TextEmbedder initialized successfully",
            model_name=self._model_name,
            embedding_dimension=self._embedding_dimension,
            model_path=str(model_dir),
            component="embedder"
        )

    @component.output_types(embedding=List[List[float]])
    def run(self, text: List[str]) -> dict[str, object]:  # type: ignore[override]
        """Return embeddings for the provided texts.

        Args:
            text: List of input texts to embed.

        Returns:
            Dict with a single key `embedding` containing the list of list[float] vectors.
        """
        with LoggedOperation("text_embedding", self._logger, input_texts=len(text)) as op:
            try:
                # Try batch processing first (more efficient)
                result = self._embedder.run(text)
                embeddings = result.get("embedding", [])
                # Check if we got back a list with the same length as input texts
                if isinstance(embeddings, list) and len(embeddings) == len(text):
                    # Ensure each embedding is a list (for proper format)
                    formatted_embeddings = []
                    for emb in embeddings:
                        if isinstance(emb, list):
                            formatted_embeddings.append(emb)
                        else:
                            # Handle single values or other formats
                            formatted_embeddings.append([emb] if isinstance(emb, (int, float)) else list(emb))
                    
                    # Add context and metrics for successful batch processing
                    op.add_context(
                        successful_embeddings=len(formatted_embeddings),
                        failed_embeddings=0,
                        processing_mode="batch"
                    )
                    
                    self._metrics.counter("embeddings_generated_total", len(formatted_embeddings), component="embedder", mode="batch", status="success")
                    
                    return {"embedding": formatted_embeddings}
            except (TypeError, ValueError, AttributeError):
                # Fallback to individual processing if batch processing fails
                pass
                
            # Individual processing fallback with better error handling
            self._logger.info("Using individual embedding processing", text_count=len(text), component="embedder")
            embeddings = []
            failed_count = 0
            
            # Create a single progress bar for all embeddings
            with tqdm(total=len(text), desc="Embedding texts", unit="text") as pbar:
                for i, single_text in enumerate(text):
                    try:
                        result = self._embedder.run(single_text)
                        embedding = result["embedding"]
                        
                        # Ensure the embedding is in the correct format (list of floats)
                        if isinstance(embedding, list):
                            embeddings.append(embedding)
                        elif isinstance(embedding, (int, float)):
                            # Handle single value embeddings
                            embeddings.append([embedding])
                        else:
                            # Try to convert to list
                            embeddings.append(list(embedding))
                            
                    except Exception as e:
                        # Log the error but continue processing other texts
                        self._logger.warning(
                            "Failed to embed individual text",
                            text_index=i,
                            text_length=len(single_text),
                            error=str(e),
                            error_type=type(e).__name__,
                            fallback_action="using_zero_vector",
                            component="embedder"
                        )
                        # Use zero vector as fallback
                        embeddings.append([0.0] * self._embedding_dimension)
                        failed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
            
            # Add final context and metrics
            successful_count = len(embeddings) - failed_count
            op.add_context(
                successful_embeddings=successful_count,
                failed_embeddings=failed_count,
                processing_mode="individual"
            )
            
            self._metrics.counter("embeddings_generated_total", successful_count, component="embedder", mode="individual", status="success")
            if failed_count > 0:
                self._metrics.counter("embeddings_failed_total", failed_count, component="embedder", mode="individual", status="failed")
            
            return {"embedding": embeddings}