"""Single file processing utilities for batch processing."""

from src.core.logging_config import get_logger, LoggedOperation
from src.core.memory_utils import log_memory_usage
from .types import ProcessingResult


def process_single_srt_file(file_path: str, pipeline) -> ProcessingResult:
    """Process a single SRT file using the pipeline.
    
    Args:
        file_path: Path to the SRT file
        pipeline: The SRT-to-Qdrant pipeline instance
        
    Returns:
        ProcessingResult with success status and metrics
    """
    logger = get_logger(__name__)
    
    try:
        with LoggedOperation("single_file_processing", logger, file_path=file_path) as op:
            # Read the SRT file content
            with open(file_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            logger.debug(
                "SRT file loaded",
                file_path=file_path,
                content_length=len(srt_content),
                component="batch_processor"
            )
            
            # Run the pipeline
            result = pipeline.run({"pre": {"srt_text": srt_content}})
            
            # Extract statistics from the result
            documents_written = 0
            if "write" in result and "stats" in result["write"]:
                documents_written = result["write"]["stats"].get("written", 0)
            
            op.add_context(
                documents_written=documents_written,
                file_size_bytes=len(srt_content)
            )
            
            logger.info(
                "SRT file processed successfully",
                file_path=file_path,
                documents_written=documents_written,
                component="batch_processor"
            )
            
            # Calculate duration manually since LoggedOperation doesn't expose duration_ms
            import time
            processing_time_ms = int((time.time() - op.start_time) * 1000)
            
            return ProcessingResult(
                file_path=file_path,
                success=True,
                documents_written=documents_written,
                processing_time_ms=processing_time_ms
            )
            
    except Exception as e:
        # Log memory stats on error for debugging
        log_memory_usage("error during file processing", logger)
        
        # Check if this is a memory-related error
        error_str = str(e).lower()
        is_memory_error = any(phrase in error_str for phrase in [
            "cuda out of memory", "out of memory", "memory", "allocation"
        ])
        
        if is_memory_error:
            logger.error(
                "Memory-related error processing SRT file",
                file_path=file_path,
                error=str(e),
                error_type=type(e).__name__,
                is_memory_error=True,
                component="batch_processor"
            )
        else:
            logger.error(
                "Failed to process SRT file",
                file_path=file_path,
                error=str(e),
                error_type=type(e).__name__,
                component="batch_processor"
            )
        
        return ProcessingResult(
            file_path=file_path,
            success=False,
            error_message=str(e)
        )