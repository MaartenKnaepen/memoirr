"""Metadata component package.

This package contains components and utilities for fetching and managing
movie metadata from sources like TMDB, Radarr, and Plex.
"""

from src.components.metadata.utilities.types import CastMember, MovieMetadata
from src.components.metadata.tmdb_client import TmdbClient
from src.components.metadata.radarr_client import RadarrClient
from src.components.metadata.plex_client import PlexClient
from src.components.metadata.metadata_service import MetadataService

__all__ = [
    "CastMember",
    "MovieMetadata",
    "TmdbClient",
    "RadarrClient",
    "PlexClient",
    "MetadataService",
]
