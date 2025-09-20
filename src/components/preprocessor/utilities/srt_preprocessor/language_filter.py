"""English language filtering utilities with proper language detection.

Uses langdetect library for accurate language identification, with fallback to ASCII heuristic.
Configurable via environment variables for flexibility.
"""
from __future__ import annotations

from typing import Iterable, List

from src.components.preprocessor.utilities.srt_preprocessor.types import CaptionUnit
from src.core.logging_config import get_logger


def is_english_text_heuristic(text: str) -> bool:
    """Return True if text appears to be English via ASCII heuristic (fallback method).

    We treat a line as English if at least the configured threshold of its characters are ASCII.
    This is used as a fallback when langdetect fails or is disabled.
    """
    if not text or not text.strip():
        return False

    from src.core.config import get_settings
    settings = get_settings()
    
    ascii_chars = sum(1 for ch in text if ord(ch) < settings.ascii_char_upper_limit)
    ratio = ascii_chars / max(1, len(text))
    return ratio >= settings.english_ascii_threshold


def is_english_text_langdetect(text: str) -> bool:
    """Return True if text is detected as English using langdetect library.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if text is detected as English with sufficient confidence, False otherwise
        
    Raises:
        ImportError: If langdetect is not available
        Exception: If langdetect fails (caller should handle gracefully)
    """
    if not text or not text.strip():
        return False
        
    from src.core.config import get_settings
    settings = get_settings()
    
    try:
        from langdetect import detect_langs
        
        # Get language probabilities
        lang_probs = detect_langs(text)
        
        # Find English probability
        english_confidence = 0.0
        for lang_prob in lang_probs:
            if lang_prob.lang == 'en':
                english_confidence = lang_prob.prob
                break
        
        return english_confidence >= settings.langdetect_confidence_threshold
        
    except ImportError:
        raise ImportError("langdetect library not available")
    except Exception:
        # langdetect can fail on very short texts, mixed languages, etc.
        # Let the caller decide how to handle this
        raise


def is_english_text(text: str) -> bool:
    """Return True if text appears to be English using the configured detection method.
    
    Uses langdetect by default with fallback to ASCII heuristic based on configuration.
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if text is detected as English, False otherwise
    """
    if not text or not text.strip():
        return False
        
    from src.core.config import get_settings
    settings = get_settings()
    logger = get_logger(__name__)
    
    # Try langdetect if enabled
    if settings.use_langdetect:
        try:
            result = is_english_text_langdetect(text)
            logger.debug(
                "Language detection completed",
                method="langdetect",
                text_length=len(text),
                is_english=result,
                component="language_filter"
            )
            return result
            
        except ImportError:
            logger.warning(
                "langdetect library not available, falling back to ASCII heuristic",
                component="language_filter",
                recommendation="Install langdetect: pip install langdetect"
            )
            # Fall through to ASCII heuristic
            
        except Exception as e:
            logger.debug(
                "langdetect failed for text, using fallback method",
                error=str(e),
                error_type=type(e).__name__,
                text_length=len(text),
                text_preview=text[:50] + "..." if len(text) > 50 else text,
                component="language_filter"
            )
            
            # If fallback is disabled, treat as non-English
            if not settings.langdetect_fallback_to_ascii:
                logger.debug(
                    "Fallback to ASCII disabled, treating as non-English",
                    component="language_filter"
                )
                return False
            # Otherwise fall through to ASCII heuristic
    
    # Use ASCII heuristic (either as primary method or fallback)
    result = is_english_text_heuristic(text)
    method = "ascii_fallback" if settings.use_langdetect else "ascii_primary"
    
    logger.debug(
        "Language detection completed",
        method=method,
        text_length=len(text),
        is_english=result,
        component="language_filter"
    )
    
    return result


def filter_english_captions(captions: Iterable[CaptionUnit]) -> List[CaptionUnit]:
    """Keep only English lines within each caption; drop caption if none remain.

    The check is line-level; a bilingual caption will retain English lines
    and drop non-English ones. Captions without any English lines are removed.
    
    Uses the configured language detection method (langdetect with ASCII fallback by default).
    """
    from src.core.logging_config import MetricsLogger, get_logger
    
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)
    
    kept: List[CaptionUnit] = []
    total_captions = 0
    total_lines = 0
    kept_lines = 0
    
    for cap in captions:
        total_captions += 1
        total_lines += len(cap.lines)
        
        en_lines = [ln for ln in cap.lines if is_english_text(ln)]
        kept_lines += len(en_lines)
        
        if en_lines:
            kept.append(
                CaptionUnit(
                    caption_index=cap.caption_index,
                    start_ms=cap.start_ms,
                    end_ms=cap.end_ms,
                    lines=en_lines,
                )
            )
    
    # Log filtering results
    dropped_captions = total_captions - len(kept)
    dropped_lines = total_lines - kept_lines
    
    logger.info(
        "Caption language filtering completed",
        total_captions=total_captions,
        kept_captions=len(kept),
        dropped_captions=dropped_captions,
        total_lines=total_lines,
        kept_lines=kept_lines,
        dropped_lines=dropped_lines,
        component="language_filter"
    )
    
    # Record metrics
    metrics.counter("captions_language_filtered_total", total_captions, component="language_filter")
    metrics.counter("captions_kept_after_language_filter", len(kept), component="language_filter", status="kept")
    metrics.counter("captions_dropped_after_language_filter", dropped_captions, component="language_filter", status="dropped")
    metrics.counter("lines_language_filtered_total", total_lines, component="language_filter")
    metrics.counter("lines_kept_after_language_filter", kept_lines, component="language_filter", status="kept")
    metrics.counter("lines_dropped_after_language_filter", dropped_lines, component="language_filter", status="dropped")
    
    return kept
