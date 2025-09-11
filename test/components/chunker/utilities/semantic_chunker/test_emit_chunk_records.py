import json
from src.components.chunker.utilities.semantic_chunker.types import (
    ChunkWithTime,
    ChunkerParams,
)
from src.components.chunker.utilities.semantic_chunker.emit_chunk_records import (
    emit_chunk_records,
)


def test_emit_chunk_records_with_params_and_indices():
    chunks = [
        ChunkWithTime(text="Hello world.", start_ms=0, end_ms=1000, token_count=5, caption_indices=[1]),
        ChunkWithTime(text="How are you?", start_ms=1000, end_ms=2000, token_count=6, caption_indices=[2]),
    ]
    params = ChunkerParams(
        threshold=0.75,
        chunk_size=512,
        similarity_window=3,
        min_sentences=2,
        min_characters_per_sentence=24,
        delim=[". "],
        include_delim="prev",
        skip_window=0,
    )
    lines = list(emit_chunk_records(chunks, include_params=True, params=params))
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["text"] == "Hello world."
    assert rec0["start_ms"] == 0
    assert rec0["chunker_params"]["chunk_size"] == 512
    assert rec0["caption_indices"] == [1]
