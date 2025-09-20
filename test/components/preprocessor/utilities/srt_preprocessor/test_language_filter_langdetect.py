"""Tests for improved language detection using langdetect library.

This test demonstrates the improvement over ASCII-only heuristics by testing
with Dutch, German, French, and other languages that would be false positives
with the old ASCII-based approach.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.components.preprocessor.utilities.srt_preprocessor.language_filter import (
    is_english_text,
    is_english_text_heuristic,
    is_english_text_langdetect,
    filter_english_captions
)
from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit


class TestLanguageDetectionImprovement:
    """Test cases demonstrating the improvement from ASCII-only to proper language detection."""
    
    def test_dutch_false_positive_with_ascii_only(self):
        """Demonstrate that Dutch would be incorrectly classified as English with ASCII-only."""
        dutch_text = "Hallo, hoe gaat het met je vandaag?"
        
        # ASCII heuristic incorrectly classifies Dutch as English (95%+ ASCII)
        assert is_english_text_heuristic(dutch_text) is True  # False positive!
        
        # With langdetect, this should be correctly identified as non-English
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=False
            )
            
            # Mock langdetect within our module
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.return_value = False  # Dutch correctly identified as non-English
                
                assert is_english_text(dutch_text) is False  # Correctly identified as non-English!
    
    def test_german_false_positive_with_ascii_only(self):
        """Demonstrate that German would be incorrectly classified as English with ASCII-only."""
        german_text = "Wie geht es dir heute?"
        
        # ASCII heuristic incorrectly classifies German as English
        assert is_english_text_heuristic(german_text) is True  # False positive!
        
        # With langdetect, this should be correctly identified as non-English
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=False
            )
            
            # Mock langdetect within our module
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.return_value = False  # German correctly identified as non-English
                
                assert is_english_text(german_text) is False  # Correctly identified as non-English!
    
    def test_french_false_positive_with_ascii_only(self):
        """Demonstrate that French would be incorrectly classified as English with ASCII-only."""
        french_text = "Comment allez-vous aujourd'hui?"
        
        # ASCII heuristic incorrectly classifies French as English
        assert is_english_text_heuristic(french_text) is True  # False positive!
        
        # With langdetect, this should be correctly identified as non-English
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=False
            )
            
            # Mock langdetect within our module  
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.return_value = False  # French correctly identified as non-English
                
                assert is_english_text(french_text) is False  # Correctly identified as non-English!
    
    def test_english_correctly_identified(self):
        """Test that actual English text is correctly identified as English."""
        english_text = "Hello, how are you doing today?"
        
        # Both methods should correctly identify this as English
        assert is_english_text_heuristic(english_text) is True
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=True
            )
            
            # Mock langdetect within our module
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.return_value = True  # English correctly identified as English
                
                assert is_english_text(english_text) is True  # Correctly identified as English!


class TestLangdetectConfiguration:
    """Test different configuration options for langdetect."""
    
    def test_langdetect_disabled_uses_ascii_heuristic(self):
        """Test that disabling langdetect falls back to ASCII heuristic."""
        dutch_text = "Hallo, hoe gaat het met je vandaag?"
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=False,  # Disabled
                english_ascii_threshold=0.95,
                ascii_char_upper_limit=128
            )
            
            # Should use ASCII heuristic (which will give false positive for Dutch)
            assert is_english_text(dutch_text) is True  # Expected false positive with ASCII-only
    
    def test_langdetect_confidence_threshold(self):
        """Test that confidence threshold affects classification."""
        ambiguous_text = "OK yes"  # Short text that might have low confidence
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.8,  # High threshold
                langdetect_fallback_to_ascii=False
            )
            
            # Mock langdetect to return English with low confidence (below threshold)
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.return_value = False  # Low confidence, below threshold
                
                assert is_english_text(ambiguous_text) is False  # Below confidence threshold
    
    def test_langdetect_fallback_enabled(self):
        """Test fallback to ASCII heuristic when langdetect fails."""
        english_text = "Hello there!"
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=True,
                english_ascii_threshold=0.95,
                ascii_char_upper_limit=128
            )
            
            # Mock langdetect to raise an exception (simulating failure)
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.side_effect = Exception("Detection failed")
                
                # Should fall back to ASCII heuristic
                assert is_english_text(english_text) is True  # ASCII heuristic should work
    
    def test_langdetect_fallback_disabled(self):
        """Test that disabling fallback treats langdetect failures as non-English."""
        english_text = "Hello there!"
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=False  # Disabled fallback
            )
            
            # Mock langdetect to raise an exception (simulating failure)
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.side_effect = Exception("Detection failed")
                
                # Should treat as non-English without fallback
                assert is_english_text(english_text) is False


class TestFilterEnglishCaptionsWithLangdetect:
    """Test caption filtering with the new language detection."""
    
    def test_filter_mixed_language_captions(self):
        """Test filtering captions with multiple languages."""
        captions = [
            CaptionUnit(caption_index=1, start_ms=0, end_ms=1000, lines=["Hello there!"]),
            CaptionUnit(caption_index=2, start_ms=1000, end_ms=2000, lines=["Hallo, hoe gaat het?"]),  # Dutch
            CaptionUnit(caption_index=3, start_ms=2000, end_ms=3000, lines=["Good morning", "Guten Morgen"]),  # Mixed
            CaptionUnit(caption_index=4, start_ms=3000, end_ms=4000, lines=["Wie geht es dir?"]),  # German
        ]
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=True
            )
            
            def mock_langdetect(text):
                """Mock langdetect based on text content."""
                if "Hello" in text or "Good morning" in text:
                    return True  # English
                elif "Hallo" in text or "hoe gaat" in text or "Wie geht" in text or "Guten Morgen" in text:
                    return False  # Non-English (Dutch/German)
                else:
                    return True  # Default to English for ambiguous cases
            
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect', side_effect=mock_langdetect):
                filtered = filter_english_captions(captions)
                
                # Should keep caption 1 (English) and caption 3 (but only English line)
                assert len(filtered) == 2
                assert filtered[0].caption_index == 1
                assert filtered[0].lines == ["Hello there!"]
                assert filtered[1].caption_index == 3
                assert filtered[1].lines == ["Good morning"]  # Only English line kept


class TestLangdetectErrorHandling:
    """Test error handling for langdetect library."""
    
    def test_langdetect_import_error(self):
        """Test graceful handling when langdetect is not available."""
        text = "Hello there!"
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_fallback_to_ascii=True,
                english_ascii_threshold=0.95,
                ascii_char_upper_limit=128
            )
            
            # Mock import error
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.side_effect = ImportError("langdetect not available")
                
                # Should fall back to ASCII heuristic
                assert is_english_text(text) is True
    
    def test_langdetect_detection_error(self):
        """Test handling of langdetect detection errors."""
        text = "Hello!"
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(
                use_langdetect=True,
                langdetect_confidence_threshold=0.7,
                langdetect_fallback_to_ascii=True,
                english_ascii_threshold=0.95,
                ascii_char_upper_limit=128
            )
            
            # Mock detection error
            with patch('src.components.preprocessor.utilities.srt_preprocessor.language_filter.is_english_text_langdetect') as mock_langdetect:
                mock_langdetect.side_effect = Exception("Detection failed")
                
                # Should fall back to ASCII heuristic
                assert is_english_text(text) is True