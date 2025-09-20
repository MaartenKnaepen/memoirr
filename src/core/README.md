
# Core

This directory contains the foundational utilities and configuration that support the entire Memoirr application.

## Modules

### config.py
Centralized configuration management using Pydantic Settings. Loads configuration from environment variables and .env files, providing type-safe access to all application settings including:
- Model paths and parameters
- Processing thresholds
- API credentials
- Component-specific configurations

### logging_config.py
Structured logging setup with JSON output, metrics collection, and operation tracking. Provides:
- Standardized log format with context
- Metrics logging (counters, histograms)
- Operation timing and tracing

### model_utils.py
Utilities for working with local AI models including:
- Model path resolution
- Model loading helpers
- File system operations for model directories

These core modules are imported throughout the application to provide consistent configuration, logging, and utility functions.