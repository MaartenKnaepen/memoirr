from unittest.mock import patch
from src.components.preprocessor.utilities.srt_preprocessor.srt_processing import srt_preprocess_text


def test_srt_preprocess_text_end_to_end_counts():
    srt_text = (
        "1\n00:00:01,000 --> 00:00:02,000\n- Hello!\n[laughs]\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\n¿Dónde está la biblioteca?\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n<b>Hi again</b>\n"
    )
    
    # Mock language detection to return True for English, False for Spanish
    def mock_is_english(text):
        return "biblioteca" not in text  # Spanish text should be filtered out
    
    with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text', side_effect=mock_is_english):
        captions, stats = srt_preprocess_text(srt_text)
        texts = [c.lines[0] for c in captions]
        # Non-English second caption dropped; first cleaned; third cleaned
        assert texts == ["Hello!", "Hi again"]
        assert stats.total_captions == 3
        assert stats.kept == 2
        assert stats.dropped_non_english == 1
