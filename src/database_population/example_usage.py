"""Example usage of the batch SRT processing functionality.

This script demonstrates how to use the batch processor to populate
a Qdrant database with SRT files from a directory.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .batch_processor import process_srt_directory
from src.core.logging_config import get_logger


def main():
    """Example usage of batch SRT processing."""
    logger = get_logger(__name__)
    
    # Example directory path - adjust as needed
    srt_directory = "/path/to/your/srt/files"
    
    # Check if directory exists
    if not os.path.exists(srt_directory):
        logger.error(f"Directory does not exist: {srt_directory}")
        print(f"Please update the srt_directory variable to point to a valid directory with SRT files.")
        return
    
    try:
        # Process SRT files, adding to existing database
        print(f"Processing SRT files in: {srt_directory}")
        print("Mode: Adding to existing database (overwrite=False)")
        
        result = process_srt_directory(srt_directory, overwrite=False)
        
        # Print results
        print(f"\nBatch Processing Results:")
        print(f"Total files found: {result.total_files}")
        print(f"Successfully processed: {result.successful_files}")
        print(f"Failed: {result.failed_files}")
        print(f"Total documents written: {result.total_documents_written}")
        print(f"Processing time: {result.processing_time_ms:.2f} ms")
        
        if result.failed_files > 0:
            print(f"\nFailed files:")
            for file_result in result.file_results:
                if not file_result.success:
                    print(f"  - {file_result.file_path}: {file_result.error_message}")
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"Error: {e}")


def example_with_overwrite():
    """Example of processing with database overwrite."""
    logger = get_logger(__name__)
    
    srt_directory = "/path/to/your/srt/files"
    
    if not os.path.exists(srt_directory):
        logger.error(f"Directory does not exist: {srt_directory}")
        return
    
    try:
        print(f"Processing SRT files in: {srt_directory}")
        print("Mode: Overwriting existing database (overwrite=True)")
        
        result = process_srt_directory(srt_directory, overwrite=True)
        
        print(f"\nResults: {result.successful_files}/{result.total_files} files processed successfully")
        print(f"Documents written: {result.total_documents_written}")
        
    except Exception as e:
        logger.error(f"Batch processing with overwrite failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()