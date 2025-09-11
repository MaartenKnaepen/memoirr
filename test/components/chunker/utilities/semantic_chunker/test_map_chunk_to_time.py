from src.components.chunker.utilities.semantic_chunker.types import CaptionJson, ChunkSpan
from src.components.chunker.utilities.semantic_chunker.build_text_and_spans import (
    build_text_and_spans,
)
from src.components.chunker.utilities.semantic_chunker.map_chunk_to_time import (
    map_chunk_span_to_time,
)


def test_map_chunk_span_to_time_overlap_two_captions():
    caps = [
        CaptionJson(text="AAA", start_ms=0, end_ms=1000, caption_index=1),
        CaptionJson(text="BBB", start_ms=1200, end_ms=2200, caption_index=2),
    ]
    text, spans = build_text_and_spans(caps)
    # Span overlapping end of AAA and start of BBB, including separator
    start = len("AA")
    end = len("AAA BBB")
    ch = ChunkSpan(text=text[start:end], start_index=start, end_index=end, token_count=3)
    mapped = map_chunk_span_to_time(ch, spans)
    assert mapped.start_ms == 0
    assert mapped.end_ms == 2200
    assert mapped.caption_indices == [1, 2]
