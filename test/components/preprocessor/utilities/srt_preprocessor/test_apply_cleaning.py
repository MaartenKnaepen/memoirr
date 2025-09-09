from components.preprocessor.utilities.srt_preprocessor.apply_cleaning import clean_and_collapse_captions
from components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_clean_and_collapse_captions_basic():
    caps = [CaptionUnit(caption_index=1, start_ms=0, end_ms=1000, lines=["- Hello", "[noise]"]) ]
    out = clean_and_collapse_captions(caps)
    assert out[0].lines == ["Hello"]
