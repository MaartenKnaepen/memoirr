"""Orchestration logic for metadata resolution.

This module coordinates Plex and TMDB clients to resolve a file path to complete
movie metadata. It implements a "Plex-Only" strategy where files must exist in
Plex to be indexed.

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from src.core.logging_config import get_logger, LoggedOperation
from src.components.metadata.tmdb_client import TmdbClient
from src.components.metadata.plex_client import PlexClient
from src.components.metadata.utilities.types import MovieMetadata


logger = get_logger(__name__)


def orchestrate_metadata(
    file_path: str,
    *,
    plex_client: PlexClient,
    tmdb_client: TmdbClient,
    top_cast: int = 20,
) -> MovieMetadata:
    """Resolve a file path to complete movie metadata.

    This function implements a "Plex-Only" strategy:
    1. Find the movie in Plex by file path (fail fast if not found).
    2. Use TMDB to fetch complete metadata (cast, genres, overview).
    3. Enrich with Plex rating_key for deep-linking.

    Args:
        file_path: Absolute file path to the video file.
        plex_client: Configured PlexClient instance.
        tmdb_client: Configured TmdbClient instance.
        top_cast: Maximum number of cast members to retrieve (default: 20).

    Returns:
        Complete MovieMetadata object with cast, genres, and plex_rating_key.

    Raises:
        ValueError: If the file is not found in Plex library.
        ConnectionError: If metadata services are unreachable.
        RuntimeError: If API requests fail.
    """
    with LoggedOperation("orchestrate_metadata", logger, file_path=file_path) as op:
        # Step 1: Look up in Plex (fail fast if not found)
        plex_rating_key, tmdb_id = _try_plex_lookup(file_path, plex_client)

        if tmdb_id is None:
            logger.error(
                "File not found in Plex library",
                file_path=file_path,
            )
            raise ValueError(f"File not found in Plex library: {file_path}")

        logger.info(
            "Found movie in Plex",
            file_path=file_path,
            plex_rating_key=plex_rating_key,
            tmdb_id=tmdb_id,
        )

        # Step 2: Fetch complete metadata from TMDB
        metadata = tmdb_client.get_movie_metadata_by_id(tmdb_id, top_cast=top_cast)

        # Step 3: Enrich with Plex rating_key for deep-linking
        enriched_metadata = replace(
            metadata,
            plex_rating_key=plex_rating_key,
        )

        op.add_context(
            tmdb_id=tmdb_id,
            plex_rating_key=plex_rating_key,
            title=enriched_metadata.title,
        )

        return enriched_metadata


def _try_plex_lookup(
    file_path: str,
    plex_client: PlexClient,
) -> Tuple[str | None, int | None]:
    """Attempt to find a movie in Plex by file path.

    Args:
        file_path: Absolute file path to search for.
        plex_client: Configured PlexClient instance.

    Returns:
        Tuple of (plex_rating_key, tmdb_id). Both may be None if not found.

    Raises:
        ConnectionError: If Plex server is unreachable.
        RuntimeError: If Plex API request fails.
    """
    rating_key = plex_client.find_by_file_path(file_path)
    if rating_key is None:
        logger.debug("Movie not found in Plex by file path", file_path=file_path)
        return None, None

    # Get full metadata to extract TMDB ID
    plex_metadata = plex_client.get_metadata(rating_key)
    return rating_key, plex_metadata.tmdb_id
