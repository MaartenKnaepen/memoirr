"""Map chunk character spans back to time ranges via caption spans."""
from __future__ import annotations

from typing import List, Optional

from src.components.chunker.utilities.semantic_chunker.types import ChunkSpan, ChunkWithTime, Span


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return (a_end > b_start) and (a_start < b_end)


def map_chunk_span_to_time(
    chunk: ChunkSpan,
    spans: List[Span],
    *,
    include_indices: bool = True,
) -> ChunkWithTime:
    """Map a chunk span to [start_ms, end_ms] using overlapping caption spans.

    Args:
        chunk: Chunk span with [start_index, end_index).
        spans: Per-caption spans built over the concatenated text.
        include_indices: Whether to include caption_indices for traceability.

    Returns:
        ChunkWithTime with computed start_ms, end_ms, and optional caption_indices.
    """
    cs, ce = chunk.start_index, chunk.end_index
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    indices: List[int] = []

    for sp in spans:
        if _overlaps(cs, ce, sp.start, sp.end):
            if start_ms is None or sp.start_ms < start_ms:
                start_ms = sp.start_ms
            if end_ms is None or sp.end_ms > end_ms:
                end_ms = sp.end_ms
            if include_indices:
                indices.append(sp.caption_index)
        # optimization: if sp.start >= ce and we already found overlaps, we can break
        if sp.start >= ce and (start_ms is not None):
            break

    if start_ms is None or end_ms is None:
        # No overlaps found; fallback to zero duration at cs (should not happen if spans cover text)
        start_ms = 0
        end_ms = 0
        indices = []

    indices = sorted(set(indices)) if include_indices else None  # type: ignore[assignment]

    return ChunkWithTime(
        text=chunk.text,
        start_ms=start_ms,
        end_ms=end_ms,
        token_count=chunk.token_count,
        caption_indices=indices,  # type: ignore[arg-type]
    )
