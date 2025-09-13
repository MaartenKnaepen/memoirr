"""Haystack component to embed text using a local sentence-transformers model.

This component aligns with Memoirr's standards:
- Uses Haystack as the abstraction layer for RAG pipelines.
- Reads configuration from src.core.config.Settings (model name, device, dimension).
- Self-hosted, in-process model loaded from the local models/ folder.

It exposes a single input socket `text` and a single output `embedding`,
following the simple, JSON-serializable socket guidance in documentation/qwen.md.
"""

from pathlib import Path

from haystack import component
from haystack.components.embedders import SentenceTransformersTextEmbedder

from src.core.config import get_settings


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
        # Resolve a local sentence-transformers-compatible folder. Prefer models/<name>,
        # but also search recursively for a folder whose terminal name matches (case-insensitive).
        root = Path("models")
        model_dir = root / self._model_name
        if not model_dir.exists():
            target = self._model_name.split("/")[-1].lower()
            candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == target]
            if candidates:
                model_dir = candidates[0]
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

    @component.output_types(embedding=list)
    def run(self, text: str) -> dict[str, object]:  # type: ignore[override]
        """Return an embedding for the provided text.

        Args:
            text: The input text to embed.

        Returns:
            Dict with a single key `embedding` containing the list[float] vector.
        """
        result = self._embedder.run(text)
        # Haystack embedders typically return {"embedding": list[float]}
        return {"embedding": result["embedding"]}
