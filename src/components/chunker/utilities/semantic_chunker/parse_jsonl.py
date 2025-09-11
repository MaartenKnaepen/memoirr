"""Parse and validate cleaned caption JSONL lines into CaptionJson records."""
from __future__ import annotations

import json
from typing import Iterable, List

from src.components.chunker.utilities.semantic_chunker.types import CaptionJson


def parse_jsonl_lines(lines: Iterable[str], *, fail_fast: bool = True) -> List[CaptionJson]:
    """Parse JSONL lines into CaptionJson records.

    Args:
        lines: Iterable of JSON strings.
        fail_fast: If True, raise ValueError on invalid lines; else skip.

    Returns:
        List of CaptionJson records parsed from valid lines.
    """
    out: List[CaptionJson] = []
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            text = data["text"]
            start_ms = int(data["start_ms"])
            end_ms = int(data["end_ms"])
            caption_index = int(data["caption_index"])
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            out.append(
                CaptionJson(
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    caption_index=caption_index,
                )
            )
        except Exception as e:  # noqa: BLE001 - controlled gate
            if fail_fast:
                raise ValueError(f"Invalid JSONL line at index {i}: {e}") from e
            # else skip invalid lines
            continue
    return out
