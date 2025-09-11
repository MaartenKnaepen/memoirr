"""Convert cleaned captions into JSONL-ready dicts or strings."""
from __future__ import annotations

import json
from typing import Iterable, Iterator

from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit, JsonRecord


def to_json_records(captions: Iterable[CaptionUnit]) -> Iterator[JsonRecord]:
    """Yield minimal JsonRecord entries from cleaned captions."""
    for cap in captions:
        text = cap.lines[0] if cap.lines else ""
        yield JsonRecord(
            text=text,
            start_ms=cap.start_ms,
            end_ms=cap.end_ms,
            caption_index=cap.caption_index,
        )


def to_jsonl_lines(captions: Iterable[CaptionUnit]) -> Iterator[str]:
    """Yield JSONL lines as strings for streaming writes."""
    for rec in to_json_records(captions):
        yield json.dumps(
            {
                "text": rec.text,
                "start_ms": rec.start_ms,
                "end_ms": rec.end_ms,
                "caption_index": rec.caption_index,
            },
            ensure_ascii=False,
        )
