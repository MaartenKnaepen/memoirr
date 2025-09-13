"""Parse SRT text into caption units.

Uses the `srt` library to robustly parse subtitle blocks.
"""
from __future__ import annotations

from typing import List

import srt

from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit
from src.components.preprocessor.utilities.srt_preprocessor.exceptions import SRTParseError


def parse_srt_text(srt_text: str) -> List[CaptionUnit]:
    """Parse raw SRT text into a list of ``CaptionUnit`` objects.

    Args:
        srt_text: The raw contents of an SRT file.

    Returns:
        A list of caption units with indexes, timings (in ms), and raw lines.

    Raises:
        SRTParseError: If parsing fails due to malformed SRT content.
    """
    try:
        subs = list(srt.parse(srt_text))
    except Exception as exc:  # pylint: disable=broad-except
        raise SRTParseError("Failed to parse SRT text") from exc

    units: List[CaptionUnit] = []
    for sub in subs:
        from src.core.config import get_settings
        settings = get_settings()
        
        start_ms = int(sub.start.total_seconds() * settings.seconds_to_milliseconds_factor)
        end_ms = int(sub.end.total_seconds() * settings.seconds_to_milliseconds_factor)
        # Split on any newline to preserve original lines separate from timecodes
        lines = [line for line in str(sub.content).splitlines() if line.strip()]
        units.append(
            CaptionUnit(
                caption_index=sub.index,
                start_ms=start_ms,
                end_ms=end_ms,
                lines=lines,
            )
        )
    return units
