from src.components.preprocessor.utilities.srt_preprocessor.drop_empty import drop_empty_or_noise
from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_drop_empty_or_noise():
    caps = [
        CaptionUnit(caption_index=1, start_ms=0, end_ms=1000, lines=[""]),
        CaptionUnit(caption_index=2, start_ms=0, end_ms=1000, lines=["a"]),
        CaptionUnit(caption_index=3, start_ms=0, end_ms=1000, lines=["ok"]),
    ]
    kept = drop_empty_or_noise(caps, min_len=2)
    assert [c.caption_index for c in kept] == [3]
