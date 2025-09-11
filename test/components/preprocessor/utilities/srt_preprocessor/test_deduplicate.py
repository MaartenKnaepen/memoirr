from src.components.preprocessor.utilities.srt_preprocessor.deduplicate import deduplicate_nearby
from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_deduplicate_nearby_drops_duplicates_within_window():
    caps = [
        CaptionUnit(caption_index=1, start_ms=1000, end_ms=1500, lines=["Hello"]),
        CaptionUnit(caption_index=2, start_ms=1500, end_ms=2000, lines=["Hello"]),
        CaptionUnit(caption_index=3, start_ms=2600, end_ms=3000, lines=["Hello"]),
    ]
    kept = deduplicate_nearby(caps, window_ms=1000)
    # The second is within 500ms of first so dropped; third is 1100ms after second kept -> kept
    assert [c.caption_index for c in kept] == [1, 3]
