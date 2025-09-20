# Tools

This directory contains utility scripts and tools for development, testing, and maintenance of the Memoirr application.

## Files

### validate_config.py
A utility script that validates the application configuration by:
- Checking required environment variables
- Verifying model directories exist
- Validating Qdrant connection settings
- Ensuring all required configuration values are present and of the correct type

This tool can be run during development or deployment to catch configuration issues early.