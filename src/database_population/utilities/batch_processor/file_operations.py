"""File operation utilities for batch processing."""

import os
import glob
from typing import List

from src.core.logging_config import get_logger


def find_srt_files(directory_path: str) -> List[str]:
    """Find all SRT files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of paths to SRT files
        
    Raises:
        ValueError: If directory_path does not exist or is not a directory
    """
    logger = get_logger(__name__)
    
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Use glob to find all .srt files recursively
    pattern = os.path.join(directory_path, "**", "*.srt")
    srt_files = glob.glob(pattern, recursive=True)
    
    logger.info(
        "SRT file discovery completed",
        directory=directory_path,
        files_found=len(srt_files),
        component="batch_processor"
    )
    
    return sorted(srt_files)