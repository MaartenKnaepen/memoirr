"""Data types for batch processing functionality."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Result of processing a single SRT file."""
    file_path: str
    success: bool
    documents_written: int = 0
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None


@dataclass
class BatchProcessingResult:
    """Result of batch processing multiple SRT files."""
    total_files: int
    successful_files: int
    failed_files: int
    total_documents_written: int
    file_results: List[ProcessingResult]
    processing_time_ms: float