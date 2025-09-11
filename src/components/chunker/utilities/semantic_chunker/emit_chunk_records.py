"""Emit JSONL lines for chunks with optional params and traceability."""
from __future__ import annotations

import json
from typing import Iterable, Iterator

from src.components.chunker.utilities.semantic_chunker.types import (
    ChunkRecord,
    ChunkWithTime,
    ChunkerParams,
)


def emit_chunk_records(
    chunks: Iterable[ChunkWithTime],
    *,
    include_params: bool,
    params: ChunkerParams,
) -> Iterator[str]:
    """Yield JSONL lines for chunk records according to the project schema.

    Args:
        chunks: Mapped chunks with time ranges.
        include_params: Whether to include chunker_params in each record.
        params: Parameters to include if include_params is True.

    Yields:
        JSONL strings for each chunk.
    """
    for ch in chunks:
        record = ChunkRecord(
            text=ch.text,
            start_ms=ch.start_ms,
            end_ms=ch.end_ms,
            token_count=ch.token_count,
            caption_indices=ch.caption_indices,
            chunker_params=(
                {
                    "threshold": params.threshold,
                    "chunk_size": params.chunk_size,
                    "similarity_window": params.similarity_window,
                    "min_sentences": params.min_sentences,
                    "min_characters_per_sentence": params.min_characters_per_sentence,
                    "delim": params.delim,
                    "include_delim": params.include_delim,
                    "skip_window": params.skip_window,
                }
                if include_params
                else None
            ),
        )
        yield json.dumps(record.__dict__, ensure_ascii=False)
