"""Build a concatenated text string and caption spans for mapping back to time."""
from __future__ import annotations

from typing import List

from src.components.chunker.utilities.semantic_chunker.types import CaptionJson, Span


def build_text_and_spans(
    captions: List[CaptionJson], *, separator: str = " "
) -> tuple[str, List[Span]]:
    """Concatenate caption texts and compute per-caption character spans.

    The span indices are [start, end) with end exclusive. A single separator is
    inserted between captions (not after the last). Empty texts are permitted but
    still produce a span of length 0.

    Args:
        captions: List of cleaned caption records.
        separator: String inserted between captions during concatenation.

    Returns:
        A tuple of (concatenated_text, spans) where spans[i] corresponds to
        captions[i].
    """
    parts: List[str] = []
    spans: List[Span] = []
    cursor = 0

    for idx, c in enumerate(captions):
        start = cursor
        parts.append(c.text)
        cursor += len(c.text)
        end = cursor
        spans.append(
            Span(
                start=start,
                end=end,
                start_ms=c.start_ms,
                end_ms=c.end_ms,
                caption_index=c.caption_index,
            )
        )
        if idx < len(captions) - 1:
            parts.append(separator)
            cursor += len(separator)

    return "".join(parts), spans
