"""Core utilities and configuration for Memoirr.

This module provides centralized configuration management and logging setup.
Import this module early in your application to ensure proper logging configuration.
"""

from src.core.logging_config import configure_logging

# Configure logging when core module is imported
configure_logging()

__all__ = ["configure_logging"]
