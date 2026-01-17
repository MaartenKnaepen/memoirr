"""TMDB Client wrapper for fetching movie metadata.

This is a wrapper class (not a Haystack component) that provides a clean
interface for fetching movie metadata from The Movie Database (TMDB) API.

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Optional

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.tmdb_client.orchestrate_tmdb import (
    search_movie_id,
    fetch_full_metadata,
)


class TmdbClient:
    """Client for interacting with The Movie Database (TMDB) API.

    This wrapper manages API key configuration and provides high-level methods
    for fetching movie metadata including cast information.

    Attributes:
        api_key: TMDB API key from configuration.
        base_url: TMDB API base URL.
    """

    def __init__(self) -> None:
        """Initialize TMDB client with configuration from settings.

        Raises:
            ValueError: If TMDB API key is not configured.
        """
        settings = get_settings()
        self._logger = get_logger(__name__)

        self.api_key = settings.tmdb_api_key
        self.base_url = settings.tmdb_base_url

        if not self.api_key:
            self._logger.error("TMDB API key is missing in configuration")
            raise ValueError(
                "TMDB_API_KEY is required. Please set it in .env or environment variables."
            )

        self._logger.info(
            "TmdbClient initialized",
            base_url=self.base_url,
            api_key_present=bool(self.api_key),
        )

    def search_movie(self, title: str, *, year: Optional[int] = None) -> int:
        """Search for a movie by title and optional year.

        Args:
            title: Movie title to search for.
            year: Optional release year to narrow search results.

        Returns:
            TMDB movie ID of the first matching result.

        Raises:
            ValueError: If title is empty or no results found.
            RuntimeError: If API request fails.
        """
        return search_movie_id(
            title,
            year=year,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_movie_metadata(
        self,
        title: str,
        *,
        year: Optional[int] = None,
        top_cast: int = 20,
    ) -> MovieMetadata:
        """Fetch complete movie metadata by searching for title and year.

        This is a convenience method that combines search and metadata fetching.

        Args:
            title: Movie title to search for.
            year: Optional release year to narrow search results.
            top_cast: Maximum number of cast members to retrieve (default: 20).

        Returns:
            Complete MovieMetadata object with movie details and cast.

        Raises:
            ValueError: If title is empty or no results found.
            RuntimeError: If API requests fail.
        """
        self._logger.info("Getting movie metadata", title=title, year=year)

        # Search for movie ID
        tmdb_id = self.search_movie(title, year=year)

        # Fetch full metadata
        return self.get_movie_metadata_by_id(tmdb_id, top_cast=top_cast)

    def get_movie_metadata_by_id(
        self,
        tmdb_id: int,
        *,
        top_cast: int = 20,
    ) -> MovieMetadata:
        """Fetch complete movie metadata using TMDB ID.

        Args:
            tmdb_id: TMDB movie ID.
            top_cast: Maximum number of cast members to retrieve (default: 20).

        Returns:
            Complete MovieMetadata object with movie details and cast.

        Raises:
            ValueError: If tmdb_id is invalid.
            RuntimeError: If API requests fail.
        """
        return fetch_full_metadata(
            tmdb_id,
            api_key=self.api_key,
            base_url=self.base_url,
            top_cast=top_cast,
        )
