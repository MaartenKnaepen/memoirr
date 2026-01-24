"""Metadata service utilities package.

This package contains orchestration logic for the MetadataService component,
coordinating Plex, Radarr, and TMDB clients to resolve file paths to metadata.
"""

from src.components.metadata.utilities.metadata_service.orchestrate_metadata import (
    orchestrate_metadata,
)

__all__ = ["orchestrate_metadata"]
