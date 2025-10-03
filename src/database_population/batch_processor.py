"""Main batch processing functionality for populating Qdrant database with SRT files.

This module provides the main processing function that orchestrates batch processing
of SRT files using utilities from the utilities folder.
"""

from typing import List

from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.core.memory_utils import clear_gpu_memory, log_memory_usage, memory_managed_operation
from src.pipelines.srt_to_qdrant import build_srt_to_qdrant_pipeline

from .utilities.batch_processor.types import BatchProcessingResult, ProcessingResult
from .utilities.batch_processor.file_operations import find_srt_files
from .utilities.batch_processor.single_file_processor import process_single_srt_file
from .utilities.batch_processor.database_operations import clear_qdrant_database


def process_srt_directory(directory_path: str, overwrite: bool = False) -> BatchProcessingResult:
    """Process all SRT files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory containing SRT files
        overwrite: If True, clear the database before processing. If False, add to existing data.
        
    Returns:
        BatchProcessingResult with processing statistics
        
    Raises:
        ValueError: If directory_path is invalid
    """
    logger = get_logger(__name__)
    metrics = MetricsLogger(logger)
    
    with LoggedOperation("batch_srt_processing", logger, 
                        directory=directory_path, 
                        overwrite=overwrite) as batch_op:
        
        logger.info(
            "Starting batch SRT processing",
            directory=directory_path,
            overwrite=overwrite,
            component="batch_processor"
        )
        
        # Find all SRT files
        try:
            srt_files = find_srt_files(directory_path)
        except ValueError as e:
            logger.error(
                "Failed to find SRT files",
                directory=directory_path,
                error=str(e),
                component="batch_processor"
            )
            raise
        
        if not srt_files:
            logger.warning(
                "No SRT files found in directory",
                directory=directory_path,
                component="batch_processor"
            )
            return BatchProcessingResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_documents_written=0,
                file_results=[],
                processing_time_ms=0.0
            )
        
        # Clear database if overwrite is requested
        if overwrite:
            logger.info(
                "Clearing database before processing",
                component="batch_processor"
            )
            if not clear_qdrant_database():
                logger.error(
                    "Failed to clear database, continuing anyway",
                    component="batch_processor"
                )
        
        # Build the pipeline once for all files with memory management
        try:
            with memory_managed_operation("pipeline_build", clear_before=True, clear_after=False):
                pipeline = build_srt_to_qdrant_pipeline()
                log_memory_usage("pipeline built", logger)
                logger.info(
                    "Pipeline built successfully for batch processing",
                    component="batch_processor"
                )
        except Exception as e:
            logger.error(
                "Failed to build pipeline for batch processing",
                error=str(e),
                error_type=type(e).__name__,
                component="batch_processor"
            )
            raise
        
        # Process each file
        file_results: List[ProcessingResult] = []
        successful_files = 0
        total_documents_written = 0
        
        for i, file_path in enumerate(srt_files, 1):
            logger.info(
                "Processing SRT file",
                file_index=i,
                total_files=len(srt_files),
                file_path=file_path,
                component="batch_processor"
            )
            
            # Clear GPU memory before processing each file to prevent accumulation
            if i > 1:  # Don't clear before first file since we just built the pipeline
                clear_gpu_memory()
            
            log_memory_usage(f"before file {i}/{len(srt_files)}", logger)
            
            result = process_single_srt_file(file_path, pipeline)
            file_results.append(result)
            
            log_memory_usage(f"after file {i}/{len(srt_files)}", logger)
            
            if result.success:
                successful_files += 1
                total_documents_written += result.documents_written
                metrics.counter("srt_files_processed_total", 1, 
                              component="batch_processor", status="success")
                metrics.counter("documents_written_total", result.documents_written,
                              component="batch_processor")
            else:
                metrics.counter("srt_files_processed_total", 1, 
                              component="batch_processor", status="failed")
            
            # Log progress every 10 files or at the end
            if i % 10 == 0 or i == len(srt_files):
                logger.info(
                    "Batch processing progress",
                    processed_files=i,
                    total_files=len(srt_files),
                    successful_files=successful_files,
                    failed_files=i - successful_files,
                    progress_percent=round((i / len(srt_files)) * 100, 1),
                    component="batch_processor"
                )
        
        failed_files = len(srt_files) - successful_files
        
        # Add context and final metrics
        batch_op.add_context(
            total_files=len(srt_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_documents_written=total_documents_written,
            success_rate=successful_files / len(srt_files) if srt_files else 0
        )
        
        metrics.histogram("batch_size_files", len(srt_files), component="batch_processor")
        metrics.histogram("batch_success_rate", successful_files / len(srt_files) if srt_files else 0, 
                         component="batch_processor")
        
        # Calculate duration manually since LoggedOperation doesn't expose duration_ms
        import time
        processing_time_ms = int((time.time() - batch_op.start_time) * 1000)
        
        logger.info(
            "Batch SRT processing completed",
            directory=directory_path,
            total_files=len(srt_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_documents_written=total_documents_written,
            processing_time_ms=processing_time_ms,
            component="batch_processor"
        )
        
        return BatchProcessingResult(
            total_files=len(srt_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_documents_written=total_documents_written,
            file_results=file_results,
            processing_time_ms=processing_time_ms
        )