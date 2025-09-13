"""Haystack component for time-aware semantic chunking of cleaned captions.

This component accepts cleaned JSONL lines produced by the SRT preprocessor and
emits JSONL chunk lines with start/end times mapped from the original captions.

It uses chonkie.SemanticChunker with a self-hosted, in-process embedding model
that is loaded from a local folder indicated by an environment variable.

Adheres to Memoirr code standards: type hints, Google-style docstrings, SRP.
"""


from haystack import component
from typing import Dict, List, Optional
from src.components.chunker.utilities.semantic_chunker import orchestrate_chunking as orchestrate_module


DEFAULT_DELIMS = [". ", "! ", "? ", "\n"]


@component
class SemanticChunker:
    """Haystack component to chunk cleaned captions into time-aware segments.

    Args:
        threshold: Similarity threshold (0-1), percentile (1-100], or "auto".
        chunk_size: Maximum tokens per chunk.
        similarity_window: Window for similarity smoothing/estimation.
        min_sentences: Minimum sentences per chunk.
        min_characters_per_sentence: Minimum characters per sentence.
        delim: Delimiters for sentence splitting.
        include_delim: Whether to include delimiter on "prev" or "next" side.
        skip_window: Skip window for non-consecutive merging.
        include_params: Whether to include chunker_params in each JSONL record.
        include_caption_indices: Whether to include caption_indices per chunk.
        fail_fast: If True, invalid input lines raise; else skipped with warning.
    """

    def __init__(
        self,
        *,
        threshold: float | int | str | None = None,
        chunk_size: int | None = None,
        similarity_window: int | None = None,
        min_sentences: int | None = None,
        min_characters_per_sentence: int | None = None,
        delim: List[str] | str | None = None,
        include_delim: Optional[str] | None = None,
        skip_window: int | None = None,
        include_params: bool | None = None,
        include_caption_indices: bool | None = None,
        fail_fast: bool | None = None,
    ) -> None:
        from src.core.config import get_settings
        import json

        settings = get_settings()
        # Resolve values from args or .env
        self.threshold = threshold if threshold is not None else settings.chunk_threshold  # type: ignore[assignment]
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.similarity_window = similarity_window if similarity_window is not None else settings.chunk_similarity_window
        self.min_sentences = min_sentences if min_sentences is not None else settings.chunk_min_sentences
        self.min_characters_per_sentence = (
            min_characters_per_sentence if min_characters_per_sentence is not None else settings.chunk_min_characters_per_sentence
        )
        # Delim can be JSON array or plain string
        if delim is not None:
            self.delim = delim
        else:
            try:
                self.delim = json.loads(settings.chunk_delim)
            except Exception:
                self.delim = settings.chunk_delim
        self.include_delim = include_delim if include_delim is not None else settings.chunk_include_delim
        self.skip_window = skip_window if skip_window is not None else settings.chunk_skip_window
        self.include_params = include_params if include_params is not None else settings.chunk_include_params
        self.include_caption_indices = (
            include_caption_indices if include_caption_indices is not None else settings.chunk_include_caption_indices
        )
        self.fail_fast = fail_fast if fail_fast is not None else settings.chunk_fail_fast

    @component.output_types(chunk_lines=List[str], stats=dict)
    def run(self, jsonl_lines: list) -> dict[str, object]:  # type: ignore[override]
            """Run semantic chunking on cleaned caption JSONL lines.

            Args:
                jsonl_lines: List of JSONL lines (each with text, start_ms, end_ms, caption_index).

            Returns:
                Dict with:
                - chunk_lines: JSONL lines for time-aware chunks, ready for embedding/indexing
                - stats: Summary counts from the chunking run
            """
            lines, stats = orchestrate_module.orchestrate_semantic_chunking(
                jsonl_lines,
                threshold=self.threshold,
                chunk_size=self.chunk_size,
                similarity_window=self.similarity_window,
                min_sentences=self.min_sentences,
                min_characters_per_sentence=self.min_characters_per_sentence,
                delim=self.delim,
                include_delim=self.include_delim,
                skip_window=self.skip_window,
                include_params=self.include_params,
                include_caption_indices=self.include_caption_indices,
                fail_fast=self.fail_fast,
            )
            return {"chunk_lines": lines, "stats": stats}
