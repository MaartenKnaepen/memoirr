"""Memory management utilities for GPU optimization.

This module provides utilities for managing CUDA memory in PyTorch to prevent
out-of-memory errors and optimize memory usage during model inference.
"""

import os
import gc
from typing import Dict, Any, Optional
from contextlib import contextmanager

from src.core.logging_config import get_logger


def configure_pytorch_memory():
    """Configure PyTorch CUDA memory allocation settings for optimal performance."""
    logger = get_logger(__name__)
    
    # Set PyTorch CUDA allocator configuration for memory optimization
    memory_config = [
        "expandable_segments:True",      # Reduce fragmentation with expandable allocations
        "garbage_collection_threshold:0.8",  # Auto-reclaim memory at 80% usage
        "max_split_size_mb:128"          # Prevent splitting blocks larger than 128MB
    ]
    
    config_str = ",".join(memory_config)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config_str
    
    logger.info(
        "PyTorch CUDA memory configuration set",
        config=config_str,
        component="memory_management"
    )


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    logger = get_logger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory stats before clearing
            before_allocated = torch.cuda.memory_allocated()
            before_reserved = torch.cuda.memory_reserved()
            
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Run Python garbage collection
            gc.collect()
            
            # Get memory stats after clearing
            after_allocated = torch.cuda.memory_allocated()
            after_reserved = torch.cuda.memory_reserved()
            
            freed_allocated = before_allocated - after_allocated
            freed_reserved = before_reserved - after_reserved
            
            logger.debug(
                "GPU memory cleared",
                freed_allocated_mb=freed_allocated / (1024**2),
                freed_reserved_mb=freed_reserved / (1024**2),
                current_allocated_mb=after_allocated / (1024**2),
                current_reserved_mb=after_reserved / (1024**2),
                component="memory_management"
            )
    except ImportError:
        logger.warning(
            "PyTorch not available, skipping GPU memory clearing",
            component="memory_management"
        )
    except Exception as e:
        logger.warning(
            "Failed to clear GPU memory",
            error=str(e),
            error_type=type(e).__name__,
            component="memory_management"
        )


def get_memory_stats() -> Dict[str, Any]:
    """Get current GPU memory statistics.
    
    Returns:
        Dictionary containing memory statistics in MB, or empty dict if CUDA unavailable.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available and torch.cuda.device_count() > 0:
            # Convert bytes to MB for readability
            mb = 1024 ** 2
            device = torch.cuda.current_device()
            
            stats = {
                'device': device,
                'allocated_mb': torch.cuda.memory_allocated(device) / mb,
                'reserved_mb': torch.cuda.memory_reserved(device) / mb,
                'max_allocated_mb': torch.cuda.max_memory_allocated(device) / mb,
                'max_reserved_mb': torch.cuda.max_memory_reserved(device) / mb,
                'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / mb,
                'free_memory_mb': None  # Will be calculated if available
            }
            
            # Try to get free memory (may not be available in all PyTorch versions)
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(device)
                stats['free_memory_mb'] = free_mem / mb
                stats['total_memory_mb'] = total_mem / mb
                stats['utilization_percent'] = ((total_mem - free_mem) / total_mem) * 100
            except AttributeError:
                # Fallback calculation
                stats['utilization_percent'] = (stats['reserved_mb'] / stats['total_memory_mb']) * 100
            
            return stats
    except ImportError:
        pass
    except Exception:
        pass
    
    return {}


def log_memory_usage(context: str, logger: Optional[Any] = None) -> Dict[str, Any]:
    """Log current memory usage with context.
    
    Args:
        context: Description of when this logging occurs
        logger: Optional logger instance. If None, uses default logger.
        
    Returns:
        Memory statistics dictionary
    """
    if logger is None:
        logger = get_logger(__name__)
    
    stats = get_memory_stats()
    
    if stats:
        logger.info(
            f"Memory usage - {context}",
            **stats,
            component="memory_management"
        )
    else:
        logger.debug(
            f"Memory usage - {context} (CUDA not available)",
            component="memory_management"
        )
    
    return stats


@contextmanager
def memory_managed_operation(operation_name: str, clear_before: bool = False, clear_after: bool = True):
    """Context manager for memory-managed operations.
    
    Args:
        operation_name: Name of the operation for logging
        clear_before: Whether to clear memory before the operation
        clear_after: Whether to clear memory after the operation
    """
    logger = get_logger(__name__)
    
    try:
        if clear_before:
            clear_gpu_memory()
        
        log_memory_usage(f"{operation_name} - start", logger)
        yield
        log_memory_usage(f"{operation_name} - end", logger)
        
    except Exception as e:
        logger.error(
            f"Error during {operation_name}",
            error=str(e),
            error_type=type(e).__name__,
            component="memory_management"
        )
        # Log memory stats on error for debugging
        log_memory_usage(f"{operation_name} - error", logger)
        raise
    finally:
        if clear_after:
            clear_gpu_memory()


def check_memory_availability(required_mb: Optional[float] = None) -> bool:
    """Check if sufficient GPU memory is available.
    
    Args:
        required_mb: Optional minimum required memory in MB
        
    Returns:
        True if sufficient memory available, False otherwise
    """
    stats = get_memory_stats()
    
    if not stats:
        return False  # No CUDA available
    
    if required_mb is None:
        # Check if we have at least 10% free memory
        return stats.get('utilization_percent', 100) < 90
    
    free_mb = stats.get('free_memory_mb')
    if free_mb is None:
        # Fallback: estimate free memory
        free_mb = stats['total_memory_mb'] - stats['reserved_mb']
    
    return free_mb >= required_mb