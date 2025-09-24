"""Utility functions for database population operations."""

import os

from src.core.logging_config import get_logger


def validate_directory(directory_path: str) -> None:
    """Validate that a directory path exists and is accessible.
    
    Args:
        directory_path: Path to validate
        
    Raises:
        ValueError: If directory is invalid
        PermissionError: If directory is not accessible
    """
    if not directory_path:
        raise ValueError("Directory path cannot be empty")
    
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    if not os.access(directory_path, os.R_OK):
        raise PermissionError(f"Directory is not readable: {directory_path}")


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB, or 0.0 if file doesn't exist
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except (OSError, FileNotFoundError):
        return 0.0


def format_processing_summary(
    total_files: int,
    successful_files: int,
    failed_files: int,
    total_documents: int,
    processing_time_ms: float
) -> str:
    """Format a human-readable processing summary.
    
    Args:
        total_files: Total number of files processed
        successful_files: Number of successfully processed files
        failed_files: Number of failed files
        total_documents: Total documents written to database
        processing_time_ms: Total processing time in milliseconds
        
    Returns:
        Formatted summary string
    """
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
    processing_time_sec = processing_time_ms / 1000
    
    summary = f"""
Database Population Summary:
============================
Total SRT files found: {total_files}
Successfully processed: {successful_files} ({success_rate:.1f}%)
Failed: {failed_files}
Total documents written: {total_documents}
Processing time: {processing_time_sec:.2f} seconds
"""
    
    if total_files > 0:
        avg_time_per_file = processing_time_ms / total_files
        summary += f"Average time per file: {avg_time_per_file:.1f} ms\n"
    
    if total_documents > 0 and processing_time_sec > 0:
        docs_per_second = total_documents / processing_time_sec
        summary += f"Documents per second: {docs_per_second:.1f}\n"
    
    return summary


def estimate_processing_time(file_count: int, avg_file_size_mb: float) -> str:
    """Provide a rough estimate of processing time based on file count and size.
    
    Args:
        file_count: Number of files to process
        avg_file_size_mb: Average file size in MB
        
    Returns:
        Human-readable time estimate
    """
    # Very rough estimates based on typical processing speeds
    # These should be adjusted based on actual performance testing
    base_time_per_file_sec = 2.0  # Base processing time per file
    size_factor = avg_file_size_mb * 0.5  # Additional time based on file size
    
    estimated_seconds = file_count * (base_time_per_file_sec + size_factor)
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds / 60
        return f"~{minutes:.1f} minutes"
    else:
        hours = estimated_seconds / 3600
        return f"~{hours:.1f} hours"


def get_directory_stats(directory_path: str) -> dict:
    """Get statistics about SRT files in a directory.
    
    Args:
        directory_path: Path to analyze
        
    Returns:
        Dictionary with file statistics
    """
    logger = get_logger(__name__)
    
    try:
        validate_directory(directory_path)
        
        from .batch_processor.file_operations import find_srt_files
        srt_files = find_srt_files(directory_path)
        
        if not srt_files:
            return {
                "file_count": 0,
                "total_size_mb": 0.0,
                "avg_size_mb": 0.0,
                "largest_file": None,
                "smallest_file": None
            }
        
        file_sizes = [get_file_size_mb(f) for f in srt_files]
        total_size = sum(file_sizes)
        avg_size = total_size / len(file_sizes)
        
        largest_idx = file_sizes.index(max(file_sizes))
        smallest_idx = file_sizes.index(min(file_sizes))
        
        return {
            "file_count": len(srt_files),
            "total_size_mb": total_size,
            "avg_size_mb": avg_size,
            "largest_file": {
                "path": srt_files[largest_idx],
                "size_mb": file_sizes[largest_idx]
            },
            "smallest_file": {
                "path": srt_files[smallest_idx],
                "size_mb": file_sizes[smallest_idx]
            }
        }
        
    except Exception as e:
        logger.error(
            "Failed to get directory statistics",
            directory=directory_path,
            error=str(e),
            component="batch_processor_utils"
        )
        return {
            "error": str(e),
            "file_count": 0,
            "total_size_mb": 0.0,
            "avg_size_mb": 0.0
        }