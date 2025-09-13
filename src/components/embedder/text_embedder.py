"""Haystack component to embed text using a local sentence-transformers model.

This component aligns with Memoirr's standards:
- Uses Haystack as the abstraction layer for RAG pipelines.
- Reads configuration from src.core.config.Settings (model name, device, dimension).
- Self-hosted, in-process model loaded from the local models/ folder.

It exposes a single input socket `text` and a single output `embedding`,
following the simple, JSON-serializable socket guidance in documentation/qwen.md.
"""

from typing import List

from haystack import component
from haystack.components.embedders import SentenceTransformersTextEmbedder

from src.core.config import get_settings
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
            print(f"Warning: EMBEDDING_DIMENSION not set, using fallback of {settings.embedding_dimension_fallback}")
            self._embedding_dimension = settings.embedding_dimension_fallback
        
        # Resolve model directory using common utility
        try:
            model_dir = resolve_model_path(self._model_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Embedding model not found: {e}") from e
        # Initialize embedder with the resolved local path.
        # Prefer offline mode when supported by the installed Haystack version.
        try:
            self._embedder = SentenceTransformersTextEmbedder(
                model=str(model_dir),
                local_files_only=True,
            )
        except TypeError:
            # Fallback for environments (or tests) where the underlying class signature
            # doesn't accept `local_files_only` (e.g., monkeypatched fakes).
            self._embedder = SentenceTransformersTextEmbedder(model=str(model_dir))
        # Warm up so that the first embedding call in a pipeline is not cold.
        self._embedder.warm_up()

    @component.output_types(embedding=List[List[float]])
    def run(self, text: List[str]) -> dict[str, object]:  # type: ignore[override]
        """Return embeddings for the provided texts.

        Args:
            text: List of input texts to embed.

        Returns:
            Dict with a single key `embedding` containing the list of list[float] vectors.
        """
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
                return {"embedding": formatted_embeddings}
        except (TypeError, ValueError, AttributeError):
            # Fallback to individual processing if batch processing fails
            pass
        
        # Individual processing fallback with better error handling
        embeddings = []
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
                print(f"Warning: Failed to embed text {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self._embedding_dimension)
        
        return {"embedding": embeddings}