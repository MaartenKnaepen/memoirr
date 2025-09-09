import json

from components.preprocessor.utilities.srt_preprocessor.to_jsonl import to_jsonl_lines
from components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_to_jsonl_lines_emits_expected_fields():
    caps = [CaptionUnit(caption_index=5, start_ms=123, end_ms=456, lines=["Hello there"]) ]
    lines = list(to_jsonl_lines(caps))
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec == {"text": "Hello there", "start_ms": 123, "end_ms": 456, "caption_index": 5}
