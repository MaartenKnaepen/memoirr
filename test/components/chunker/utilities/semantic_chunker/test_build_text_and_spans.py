from src.components.chunker.utilities.semantic_chunker.types import CaptionJson
from src.components.chunker.utilities.semantic_chunker.build_text_and_spans import (
    build_text_and_spans,
)


def test_build_text_and_spans_basic():
    caps = [
        CaptionJson(text="Hello.", start_ms=0, end_ms=1000, caption_index=1),
        CaptionJson(text="How are you?", start_ms=1000, end_ms=2000, caption_index=2),
    ]
    text, spans = build_text_and_spans(caps)
    assert text == "Hello. How are you?"
    assert spans[0].start == 0
    assert spans[0].end == len("Hello.")
    assert spans[1].start == len("Hello.") + 1
    assert spans[1].end == len(text)
