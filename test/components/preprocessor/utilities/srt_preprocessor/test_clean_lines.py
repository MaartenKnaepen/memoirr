from src.components.preprocessor.utilities.srt_preprocessor.clean_lines import clean_caption_lines


def test_clean_caption_lines_strips_tags_and_cues_and_dashes():
    lines = ["<i>Hi</i>", "[applause]", "♪ la la ♪", "- Hello"]
    cleaned = clean_caption_lines(lines)
    assert cleaned == "Hi Hello"
