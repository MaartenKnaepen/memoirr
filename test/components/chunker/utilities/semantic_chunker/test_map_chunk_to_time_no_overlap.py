from src.components.chunker.utilities.semantic_chunker.types import CaptionJson, ChunkSpan
from src.components.chunker.utilities.semantic_chunker.build_text_and_spans import build_text_and_spans
from src.components.chunker.utilities.semantic_chunker.map_chunk_to_time import map_chunk_span_to_time


def test_map_chunk_span_to_time_no_overlap_returns_zero_range():
    caps = [
        CaptionJson(text="AAA", start_ms=100, end_ms=200, caption_index=1),
    ]
    text, spans = build_text_and_spans(caps)
    # Create a chunk completely outside the span [0, len(text)) -> choose indexes after end
    start = len(text) + 5
    end = start + 3
    ch = ChunkSpan(text="XXX", start_index=start, end_index=end, token_count=1)
    mapped = map_chunk_span_to_time(ch, spans)
    assert mapped.start_ms == 0 and mapped.end_ms == 0
    assert mapped.caption_indices == []
