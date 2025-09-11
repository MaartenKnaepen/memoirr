from src.components.preprocessor.utilities.srt_preprocessor.language_filter import filter_english_captions
from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_filter_english_captions_keeps_english_drops_non_english():
    caps = [
        CaptionUnit(caption_index=1, start_ms=0, end_ms=1000, lines=["Hello there!"]),
        CaptionUnit(caption_index=2, start_ms=1000, end_ms=2000, lines=["¿Dónde está la biblioteca?"]),
        CaptionUnit(caption_index=3, start_ms=2000, end_ms=3000, lines=["Good morning", "¿Qué tal?"]),
    ]
    filtered = filter_english_captions(caps)
    # 1st should remain, 2nd should be dropped, 3rd should keep only the English line
    assert len(filtered) == 2
    assert filtered[0].caption_index == 1 and filtered[0].lines == ["Hello there!"]
    assert filtered[1].caption_index == 3 and filtered[1].lines == ["Good morning"]
