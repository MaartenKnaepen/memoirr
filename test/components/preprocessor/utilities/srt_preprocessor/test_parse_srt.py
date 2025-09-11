from src.components.preprocessor.utilities.srt_preprocessor.parse_srt import parse_srt_text


def test_parse_srt_basic():
    srt_text = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "Hello!\n\n"
        "2\n"
        "00:00:02,500 --> 00:00:04,000\n"
        "Second line\n"
    )
    units = parse_srt_text(srt_text)
    assert len(units) == 2
    assert units[0].caption_index == 1
    assert units[0].start_ms == 1000 and units[0].end_ms == 2000
    assert units[1].caption_index == 2
    assert units[1].start_ms == 2500 and units[1].end_ms == 4000
