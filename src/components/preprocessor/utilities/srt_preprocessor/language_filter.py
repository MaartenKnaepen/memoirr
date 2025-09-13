"""Lightweight English language filtering utilities.

For now we avoid heavy dependencies. The heuristic is replaceable later.
"""
from __future__ import annotations

from typing import Iterable, List

from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit

def is_english_text_heuristic(text: str) -> bool:
    """Return True if text appears to be English via a simple ASCII heuristic.

    We treat a line as English if at least the configured threshold of its characters are ASCII.
    This is a pragmatic default for SRTs assumed to be English, while
    effectively filtering out bilingual lines with non-Latin diacritics.
    """
    if not text or not text.strip():
        return False

    from src.core.config import get_settings
    settings = get_settings()
    
    ascii_chars = sum(1 for ch in text if ord(ch) < settings.ascii_char_upper_limit)
    ratio = ascii_chars / max(1, len(text))
    return ratio >= settings.english_ascii_threshold


def filter_english_captions(captions: Iterable[CaptionUnit]) -> List[CaptionUnit]:
    """Keep only English lines within each caption; drop caption if none remain.

    The check is line-level; a bilingual caption will retain English lines
    and drop non-English ones. Captions without any English lines are removed.
    """
    kept: List[CaptionUnit] = []
    for cap in captions:
        en_lines = [ln for ln in cap.lines if is_english_text_heuristic(ln)]
        if en_lines:
            kept.append(
                CaptionUnit(
                    caption_index=cap.caption_index,
                    start_ms=cap.start_ms,
                    end_ms=cap.end_ms,
                    lines=en_lines,
                )
            )
    return kept
