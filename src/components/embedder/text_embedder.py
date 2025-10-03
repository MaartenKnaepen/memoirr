"""Haystack component to embed text using a local sentence-transformers model.

This component aligns with Memoirr's standards:
- Uses Haystack as the abstraction layer for RAG pipelines.
- Reads configuration from src.core.config.Settings (model name, device, dimension).
- Self-hosted, in-process model loaded from the local models/ folder.

It exposes a single input socket `text` and a single output `embedding`,
following the simple, JSON-serializable socket guidance in documentation/qwen.md.
"""

import os
from typing import List
from contextlib import contextmanager

from haystack.core.component import component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from tqdm import tqdm

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.core.model_utils import resolve_model_path


@contextmanager
def disable_tqdm():
    """Context manager to disable all tqdm progress bars."""
    # Save original environment
    original_disable = os.environ.get('TQDM_DISABLE', None)
    
    # Disable tqdm
    os.environ['TQDM_DISABLE'] = '1'
    
    # Also try to disable sentence-transformers progress bars
    import logging
    sentence_transformers_logger = logging.getLogger('sentence_transformers')
    original_level = sentence_transformers_logger.level
    sentence_transformers_logger.setLevel(logging.WARNING)
    
    try:
        yield
    finally:
        # Restore original environment
        if original_disable is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = original_disable
        
        # Restore logging level
        sentence_transformers_logger.setLevel(original_level)


@component
class TextEmbedder:
    """Embed text using a local sentence-transformers model.

    The model folder is resolved relative to the repository's `models/` directory using
    the `EMBEDDING_MODEL_NAME` environment variable (via Settings). This should match
    the model used by the semantic chunker to ensure consistent embedding spaces.
    
    Uses SentenceTransformersDocumentEmbedder for efficient batch processing.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        # The SentenceTransformersDocumentEmbedder accepts a `model` argument that can be
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
        # Use DocumentEmbedder for efficient batch processing of multiple texts
        # Disable progress bars to avoid individual 1/1 progress bars
        try:
            self._embedder = SentenceTransformersDocumentEmbedder(
                model=str(model_dir),
                local_files_only=True,
                show_progress_bar=False,
            )
        except TypeError:
            # Fallback for environments (or tests) where the underlying class signature
            # doesn't accept `local_files_only` or `show_progress_bar` (e.g., monkeypatched fakes).
            try:
                self._embedder = SentenceTransformersDocumentEmbedder(
                    model=str(model_dir),
                    show_progress_bar=False,
                )
            except TypeError:
                # Final fallback for minimal constructor
                self._embedder = SentenceTransformersDocumentEmbedder(model=str(model_dir))
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
                # Convert texts to Documents for batch processing with DocumentEmbedder
                documents = [Document(content=t) for t in text]
                
                # Use batch processing (more efficient)
                with disable_tqdm():
                    result = self._embedder.run(documents)
                
                # Extract embeddings from documents
                embedded_docs = result.get("documents", [])
                if len(embedded_docs) == len(text):
                    embeddings = []
                    for doc in embedded_docs:
                        if hasattr(doc, 'embedding') and doc.embedding is not None:
                            embeddings.append(doc.embedding)
                        else:
                            # Fallback to zero vector if embedding is missing
                            embeddings.append([0.0] * self._embedding_dimension)
                    
                    # Add context and metrics for successful batch processing
                    op.add_context(
                        successful_embeddings=len(embeddings),
                        failed_embeddings=0,
                        processing_mode="batch"
                    )
                    
                    self._metrics.counter("embeddings_generated_total", len(embeddings), component="embedder", mode="batch", status="success")
                    
                    return {"embedding": embeddings}
            except (TypeError, ValueError, AttributeError) as e:
                # Log the error for debugging but continue with fallback
                self._logger.warning(
                    "Batch processing failed, using individual fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    component="embedder"
                )
                
            # Individual processing fallback with better error handling
            self._logger.info("Using individual embedding processing", text_count=len(text), component="embedder")
            embeddings = []
            failed_count = 0
            
            # Create a single progress bar for all embeddings while disabling individual ones
            with tqdm(total=len(text), desc="Embedding texts", unit="text") as pbar:
                for i, single_text in enumerate(text):
                    try:
                        # Convert single text to Document for DocumentEmbedder
                        single_doc = [Document(content=single_text)]
                        
                        # Disable all tqdm progress bars during individual embedding
                        with disable_tqdm():
                            result = self._embedder.run(single_doc)
                        
                        # Extract embedding from the single document result
                        embedded_docs = result.get("documents", [])
                        if embedded_docs and hasattr(embedded_docs[0], 'embedding') and embedded_docs[0].embedding is not None:
                            embeddings.append(embedded_docs[0].embedding)
                        else:
                            # Fallback to zero vector if embedding is missing
                            embeddings.append([0.0] * self._embedding_dimension)
                            failed_count += 1
                            
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