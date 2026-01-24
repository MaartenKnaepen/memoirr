"""Haystack component for resolving file paths to movie metadata.

This component integrates TmdbClient and PlexClient to resolve video file paths
to rich metadata. It implements a "Plex-Only" strategy where files must exist
in Plex to be indexed (enabling deep-link playback).

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Dict, Optional

from haystack import component

from src.core.config import get_settings
from src.core.logging_config import get_logger, MetricsLogger
from src.components.metadata.tmdb_client import TmdbClient
from src.components.metadata.plex_client import PlexClient
from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.metadata_service.orchestrate_metadata import (
    orchestrate_metadata,
)


@component
class MetadataService:
    """Haystack component for resolving file paths to movie metadata.

    This component coordinates Plex and TMDB to provide complete movie metadata
    from a file path. It implements a "Plex-Only" strategy: if a file isn't in
    Plex, it won't be indexed (we can't deep-link to it anyway).

    Args:
        top_cast: Maximum number of cast members to retrieve (default: 20).
        plex_url: Optional override for Plex server URL.
        plex_token: Optional override for Plex authentication token.
    """

    def __init__(
        self,
        *,
        top_cast: Optional[int] = None,
        plex_url: Optional[str] = None,
        plex_token: Optional[str] = None,
    ) -> None:
        """Initialize MetadataService with configured clients.

        Args:
            top_cast: Maximum number of cast members to retrieve.
            plex_url: Optional override for Plex server URL.
            plex_token: Optional override for Plex authentication token.

        Raises:
            ValueError: If required API keys or URLs are not configured.
        """
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)

        # Configuration
        self.top_cast = top_cast if top_cast is not None else 20

        # Initialize clients (Plex-Only strategy: no Radarr)
        self._tmdb_client = TmdbClient()
        self._plex_client = PlexClient(
            base_url=plex_url,
            token=plex_token,
        )

        self._logger.info(
            "MetadataService initialized",
            top_cast=self.top_cast,
            component="metadata_service",
        )

    @component.output_types(metadata=MovieMetadata)
    def run(self, file_path: str) -> Dict[str, MovieMetadata]:  # type: ignore[override]
        """Resolve a file path to complete movie metadata.

        This method uses a "Plex-Only" strategy:
        1. Find the movie in Plex by file path (fail fast if not found).
        2. Fetch complete metadata from TMDB.
        3. Enrich with Plex rating_key for deep-linking.

        Args:
            file_path: Absolute file path to the video file.

        Returns:
            Dict with:
            - metadata: Complete MovieMetadata object with cast, genres,
              and plex_rating_key for deep-linking.

        Raises:
            ValueError: If the file is not found in Plex library.
            ConnectionError: If metadata services are unreachable.
            RuntimeError: If API requests fail.
        """
        self._logger.info(
            "Resolving metadata for file",
            file_path=file_path,
            component="metadata_service",
        )
        self._metrics.counter("metadata_service_requests", 1)

        try:
            metadata = orchestrate_metadata(
                file_path,
                plex_client=self._plex_client,
                tmdb_client=self._tmdb_client,
                top_cast=self.top_cast,
            )

            self._metrics.counter("metadata_service_success", 1)
            self._logger.info(
                "Metadata resolved successfully",
                file_path=file_path,
                title=metadata.title,
                tmdb_id=metadata.tmdb_id,
                plex_rating_key=metadata.plex_rating_key,
                component="metadata_service",
            )

            return {"metadata": metadata}

        except ValueError as e:
            self._metrics.counter("metadata_service_not_found", 1)
            self._logger.warning(
                "File not found in Plex library",
                file_path=file_path,
                error=str(e),
                component="metadata_service",
            )
            raise

        except (ConnectionError, RuntimeError) as e:
            self._metrics.counter("metadata_service_errors", 1)
            self._logger.error(
                "Metadata resolution failed",
                file_path=file_path,
                error=str(e),
                component="metadata_service",
            )
            raise
