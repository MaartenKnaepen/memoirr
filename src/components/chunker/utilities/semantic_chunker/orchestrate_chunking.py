"""Orchestrate time-aware semantic chunking for cleaned caption JSONL.

This function composes the pure utilities to:
- parse cleaned JSONL lines into records
- build concatenated text and spans
- run chonkie SemanticChunker with a local, self-hosted embeddings model
- map chunk spans back to time ranges
- emit JSONL chunk records and summary stats

Follows the same pattern as the SRT preprocessor's orchestration.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from src.components.chunker.utilities.semantic_chunker.build_text_and_spans import (
    build_text_and_spans,
)
from src.components.chunker.utilities.semantic_chunker.emit_chunk_records import (
    emit_chunk_records,
)
from src.components.chunker.utilities.semantic_chunker.map_chunk_to_time import (
    map_chunk_span_to_time,
)
from src.components.chunker.utilities.semantic_chunker.parse_jsonl import parse_jsonl_lines
from src.components.chunker.utilities.semantic_chunker.run_semantic_chunker import (
    run_semantic_chunker,
)
from src.components.chunker.utilities.semantic_chunker.types import ChunkerParams
from src.core.config import get_settings


def orchestrate_semantic_chunking(
    jsonl_lines: Iterable[str],
    *,
    threshold: float | int | str = "auto",
    chunk_size: int = 512,
    similarity_window: int = 3,
    min_sentences: int = 2,
    min_characters_per_sentence: int = 24,
    delim: List[str] | str = (". ", "! ", "? ", "\n"),
    include_delim: str | None = "prev",
    skip_window: int = 0,
    include_params: bool = True,
    include_caption_indices: bool = True,
    fail_fast: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Run the end-to-end chunking and return (jsonl_lines, stats).

    Args:
        jsonl_lines: Cleaned caption JSONL input.
        threshold: Threshold (float [0,1], percentile (1,100], or "auto").
        chunk_size: Max tokens per chunk.
        similarity_window: Similarity smoothing window.
        min_sentences: Minimum sentences per chunk.
        min_characters_per_sentence: Minimum characters per sentence.
        delim: Delimiters.
        include_delim: Whether to include delimiter on the previous or next side.
        skip_window: Skip window for non-consecutive merging.
        include_params: Include chunker_params in output records.
        include_caption_indices: Include caption_indices in output records.
        fail_fast: Raise on invalid input line vs skip.

    Returns:
        A tuple of (chunk_lines, stats).
    """
    caps = parse_jsonl_lines(jsonl_lines, fail_fast=fail_fast)
    text, spans = build_text_and_spans(caps)

    params = ChunkerParams(
        threshold=threshold,
        chunk_size=chunk_size,
        similarity_window=similarity_window,
        min_sentences=min_sentences,
        min_characters_per_sentence=min_characters_per_sentence,
        delim=list(delim) if isinstance(delim, tuple) else delim,
        include_delim=include_delim,
        skip_window=skip_window,
    )

    settings = get_settings()

    chunks = run_semantic_chunker(
        text,
        params,
        model_name=settings.embedding_model_name,
        device=settings.device,
    )

    mapped = [
        map_chunk_span_to_time(ch, spans, include_indices=include_caption_indices)
        for ch in chunks
    ]

    out_lines = list(
        emit_chunk_records(
            mapped,
            include_params=include_params,
            params=params,
        )
    )

    stats = {
        "input_captions": len(caps),
        "output_chunks": len(mapped),
        "avg_tokens_per_chunk": (
            (sum(m.token_count for m in mapped) / len(mapped)) if mapped else 0.0
        ),
    }
    return out_lines, stats
