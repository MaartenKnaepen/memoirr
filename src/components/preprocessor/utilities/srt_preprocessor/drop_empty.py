"""Drop captions that became empty or noise after cleaning."""
from __future__ import annotations

from typing import Iterable, List

from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def drop_empty_or_noise(captions: Iterable[CaptionUnit], *, min_len: int = 1) -> List[CaptionUnit]:
    """Remove captions with empty content or too-short text.

    Args:
        captions: Iterable of caption units with ``lines`` possibly containing
            a single cleaned string.
        min_len: Minimum number of characters required to keep a caption.

    Returns:
        Filtered list of captions.
    """
    kept: List[CaptionUnit] = []
    for cap in captions:
        text = cap.lines[0] if cap.lines else ""
        if text and len(text) >= min_len:
            kept.append(cap)
    return kept
