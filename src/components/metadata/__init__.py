"""Metadata component package.

This package contains components and utilities for fetching and managing
movie metadata from sources like TMDB, Radarr, and Plex.
"""

from src.components.metadata.utilities.types import CastMember, MovieMetadata
from src.components.metadata.tmdb_client import TmdbClient

__all__ = ["CastMember", "MovieMetadata", "TmdbClient"]
