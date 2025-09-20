from unittest.mock import patch, MagicMock

from src.components.preprocessor.utilities.srt_preprocessor.language_filter import filter_english_captions
from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


def test_filter_english_captions_keeps_english_drops_non_english():
    """Test that the filter keeps English and drops non-English captions using mocked langdetect."""
    caps = [
        CaptionUnit(caption_index=1, start_ms=0, end_ms=1000, lines=["Hello there!"]),
        CaptionUnit(caption_index=2, start_ms=1000, end_ms=2000, lines=["¿Dónde está la biblioteca?"]),
        CaptionUnit(caption_index=3, start_ms=2000, end_ms=3000, lines=["Good morning", "¿Qué tal?"]),
    ]
    
    with patch('src.core.config.get_settings') as mock_settings:
        mock_settings.return_value = MagicMock(
            use_langdetect=True,
            langdetect_confidence_threshold=0.7,
            langdetect_fallback_to_ascii=True
        )
        
        def mock_detect_langs(text):
            """Mock langdetect to identify English vs Spanish text."""
            if any(word in text.lower() for word in ["hello", "good morning"]):
                return [MagicMock(lang='en', prob=0.9)]
            else:
                return [MagicMock(lang='es', prob=0.9)]
        
        with patch('langdetect.detect_langs', side_effect=mock_detect_langs):
            filtered = filter_english_captions(caps)
            
            # 1st should remain, 2nd should be dropped, 3rd should keep only the English line
            assert len(filtered) == 2
            assert filtered[0].caption_index == 1 and filtered[0].lines == ["Hello there!"]
            assert filtered[1].caption_index == 3 and filtered[1].lines == ["Good morning"]
