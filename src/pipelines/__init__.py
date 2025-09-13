"""Pipeline compositions for end-to-end processing workflows.

This module contains pipeline definitions that compose individual components
into complete processing workflows.
"""

from src.core.logging_config import get_logger

# Initialize logging for the pipelines module
logger = get_logger(__name__)
logger.info(
    "Pipelines module initialized",
    module="pipelines",
    available_pipelines=["srt_to_qdrant"]
)