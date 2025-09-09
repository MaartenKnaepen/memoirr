"""Apply line cleaning to each caption and collapse to a single string line."""
from __future__ import annotations

from typing import Iterable, List

from components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit
from components.preprocessor.utilities.srt_preprocessor.clean_lines import clean_caption_lines


def clean_and_collapse_captions(captions: Iterable[CaptionUnit]) -> List[CaptionUnit]:
    """Return new captions with cleaned, single-line content per caption.

    For each caption, its ``lines`` are cleaned and collapsed into a single
    normalized string. Captions with no surviving content are kept here and
    should be removed by a subsequent filtering stage.
    """
    cleaned_caps: List[CaptionUnit] = []
    for cap in captions:
        cleaned_text = clean_caption_lines(cap.lines)
        cleaned_caps.append(
            CaptionUnit(
                caption_index=cap.caption_index,
                start_ms=cap.start_ms,
                end_ms=cap.end_ms,
                lines=[cleaned_text] if cleaned_text else [],
            )
        )
    return cleaned_caps
