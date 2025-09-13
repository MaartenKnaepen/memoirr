"""Custom Haystack component that wraps the SRT preprocessing pipeline.

This component consumes raw SRT text and emits cleaned JSONL lines plus
summary statistics, built on top of the utilities in this package.

It follows Haystack's custom component requirements:
- @component decorator
- run() method returning a dict matching @component.output_types

Note: This focuses on text input. If you prefer file inputs, you can wrap
file reading outside the pipeline or extend the component as needed.
"""
from dataclasses import asdict
from typing import Dict, List

from haystack import component

from src.components.preprocessor.utilities.srt_preprocessor.srt_processing import srt_preprocess_text
from src.components.preprocessor.utilities.srt_preprocessor.to_jsonl import (
    to_jsonl_lines,
)
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger


@component
class SRTPreprocessor:
    """Haystack component for cleaning SRT content into JSONL-ready lines.

    Args:
        min_len: Minimum number of characters required to retain a caption.
        dedupe_window_ms: Time window in milliseconds for near-duplicate removal.
    """

    def __init__(self, *, min_len: int | None = None, dedupe_window_ms: int | None = None) -> None:
        from src.core.config import get_settings
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        self.min_len = min_len if min_len is not None else settings.pre_min_len
        self.dedupe_window_ms = dedupe_window_ms if dedupe_window_ms is not None else settings.pre_dedupe_window_ms
        
        self._logger.info(
            "SRTPreprocessor initialized",
            min_len=self.min_len,
            dedupe_window_ms=self.dedupe_window_ms,
            component="preprocessor"
        )

    @component.output_types(jsonl_lines=List[str], stats=dict)
    def run(self, srt_text: str) -> Dict[str, object]:  # type: ignore[override]
        """Run the SRT preprocessing on raw SRT text.

        Args:
            srt_text: Raw SRT content as a single string.

        Returns:
            A dict with two outputs:
            - jsonl_lines: List[str] of cleaned JSONL records (one per caption)
            - stats: A dict of summary statistics from the preprocessing run
        """
        with LoggedOperation("srt_preprocessing", self._logger, input_length=len(srt_text)) as op:
            self._logger.info(
                "Starting SRT preprocessing",
                input_size_chars=len(srt_text),
                min_len=self.min_len,
                dedupe_window_ms=self.dedupe_window_ms,
                component="preprocessor"
            )
            
            cleaned_caps, stats = srt_preprocess_text(
                srt_text, min_len=self.min_len, dedupe_window_ms=self.dedupe_window_ms
            )
            lines: List[str] = list(to_jsonl_lines(cleaned_caps))
            
            # Add context and metrics
            stats_dict = asdict(stats)
            op.add_context(
                output_lines=len(lines),
                captions_kept=stats_dict.get("kept", 0),
                captions_dropped=stats_dict.get("total", 0) - stats_dict.get("kept", 0)
            )
            
            self._metrics.counter("captions_processed_total", stats_dict.get("total", 0), component="preprocessor")
            self._metrics.counter("captions_kept_total", stats_dict.get("kept", 0), component="preprocessor", status="kept")
            self._metrics.counter("captions_dropped_total", stats_dict.get("dropped_empty", 0), component="preprocessor", reason="empty")
            self._metrics.counter("captions_dropped_total", stats_dict.get("dropped_non_english", 0), component="preprocessor", reason="non_english")
            self._metrics.counter("captions_deduped_total", stats_dict.get("deduped", 0), component="preprocessor", reason="duplicate")
            
            self._logger.info(
                "SRT preprocessing completed",
                output_lines=len(lines),
                processing_stats=stats_dict,
                component="preprocessor"
            )
            
            return {"jsonl_lines": lines, "stats": stats_dict}
