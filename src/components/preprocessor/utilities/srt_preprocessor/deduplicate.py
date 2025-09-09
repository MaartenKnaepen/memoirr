"""Remove near-duplicate captions appearing within a short time window."""
from __future__ import annotations

from typing import Iterable, List

from components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def deduplicate_nearby(
    captions: Iterable[CaptionUnit], *, window_ms: int = 1000
) -> List[CaptionUnit]:
    """Drop exact duplicate texts repeated within a short time window.

    Args:
        captions: Iterable of cleaned caption units (single-line text in lines[0]).
        window_ms: Time window in milliseconds to consider a duplicate nearby.

    Returns:
        List of captions with nearby duplicates removed.
    """
    last_text_by_bucket: dict[str, int] = {}
    kept: List[CaptionUnit] = []

    for cap in captions:
        text = cap.lines[0] if cap.lines else ""
        if not text:
            kept.append(cap)
            continue
        bucket_key = text
        last_ms = last_text_by_bucket.get(bucket_key)
        if last_ms is None or (cap.start_ms - last_ms) > window_ms:
            kept.append(cap)
            last_text_by_bucket[bucket_key] = cap.start_ms
    return kept
