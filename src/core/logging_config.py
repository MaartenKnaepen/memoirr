"""Centralized logging configuration for Memoirr.

This module sets up structured logging with JSON output for production observability.
Supports local development with human-readable format and production JSON format
for integration with Grafana, Loki, and Prometheus.

Usage:
    from src.core.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started", component="preprocessor", file_count=5)
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog

from src.core.config import get_settings


class LoggingSettings:
    """Logging-specific configuration."""
    
    def __init__(self):
        settings = get_settings()
        # Add logging settings to main config if not already present
        self.log_level: str = getattr(settings, 'log_level', 'INFO')
        self.log_format: str = getattr(settings, 'log_format', 'json')  # 'json' or 'console'
        self.log_file: Optional[str] = getattr(settings, 'log_file', None)
        self.service_name: str = getattr(settings, 'service_name', 'memoirr')
        self.environment: str = getattr(settings, 'environment', 'development')


def configure_logging() -> None:
    """Configure structured logging for the application.
    
    Sets up structlog with appropriate processors for development and production.
    Should be called once at application startup.
    """
    settings = LoggingSettings()
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        stream=sys.stdout,
        format="%(message)s",  # structlog will handle formatting
    )
    
    # Base processors that always run
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_service_info,
    ]
    
    # Environment-specific processors
    if settings.log_format == "json":
        # Production: JSON output for Loki/Grafana
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Development: Human-readable console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def _add_service_info(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service metadata to all log entries."""
    settings = LoggingSettings()
    event_dict["service"] = settings.service_name
    event_dict["environment"] = settings.environment
    return event_dict


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module.
    
    Args:
        name: Usually __name__ to identify the logging module
        
    Returns:
        Configured structured logger
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", component="chunker", items=42)
    """
    return structlog.get_logger(name)


# Context managers for operation tracking
class LoggedOperation:
    """Context manager for logging operation start/end with timing.
    
    Example:
        with LoggedOperation("srt_processing", logger, file_size=1024) as op:
            # do work
            op.add_context(captions_found=42)
        # Automatically logs completion with duration
    """
    
    def __init__(self, operation: str, logger: structlog.stdlib.BoundLogger, **context):
        self.operation = operation
        self.logger = logger
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(
            f"{self.operation}_started",
            operation=self.operation,
            **self.context
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = int((time.time() - self.start_time) * 1000)
        
        if exc_type is None:
            self.logger.info(
                f"{self.operation}_completed",
                operation=self.operation,
                duration_ms=duration_ms,
                **self.context
            )
        else:
            self.logger.error(
                f"{self.operation}_failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )
    
    def add_context(self, **kwargs):
        """Add additional context during operation."""
        self.context.update(kwargs)


# Metrics helpers for Prometheus integration
class MetricsLogger:
    """Helper for logging metrics that can be scraped by Prometheus.
    
    Example:
        metrics = MetricsLogger(get_logger(__name__))
        metrics.counter("documents_processed", 1, component="chunker")
        metrics.histogram("processing_duration_ms", 1500, operation="embedding")
    """
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self.logger = logger
        
    def counter(self, metric_name: str, value: int = 1, **labels):
        """Log a counter metric."""
        self.logger.info(
            "metric_counter",
            metric_name=metric_name,
            metric_type="counter",
            value=value,
            **labels
        )
        
    def histogram(self, metric_name: str, value: float, **labels):
        """Log a histogram metric."""
        self.logger.info(
            "metric_histogram", 
            metric_name=metric_name,
            metric_type="histogram",
            value=value,
            **labels
        )
        
    def gauge(self, metric_name: str, value: float, **labels):
        """Log a gauge metric."""
        self.logger.info(
            "metric_gauge",
            metric_name=metric_name, 
            metric_type="gauge",
            value=value,
            **labels
        )