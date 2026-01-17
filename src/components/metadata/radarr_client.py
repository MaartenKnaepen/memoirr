"""Radarr client for movie library management.

This component provides methods to interact with a Radarr server to resolve
local video files to movie metadata.

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import List, Optional

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.radarr_client.orchestrate_radarr import (
    get_all_movies,
    get_movie_by_tmdb_id as orchestrate_get_movie_by_tmdb_id,
)


class RadarrClient:
    """Client for interacting with Radarr API.

    This client provides methods to query the Radarr movie library and resolve
    file paths to movie metadata. It requires a running Radarr instance with
    API access configured.

    Args:
        radarr_url: Optional override for Radarr server URL. If not provided,
            uses RADARR_URL from settings.
        radarr_api_key: Optional override for Radarr API key. If not provided,
            uses RADARR_API_KEY from settings.
    """

    def __init__(
        self,
        *,
        radarr_url: Optional[str] = None,
        radarr_api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)

        # Load configuration
        self.radarr_url = radarr_url if radarr_url is not None else settings.radarr_url
        self.radarr_api_key = radarr_api_key if radarr_api_key is not None else settings.radarr_api_key

        # Validate required configuration
        if not self.radarr_url:
            self._logger.error("Radarr URL is not configured")
            raise ValueError(
                "Radarr URL is required. Set RADARR_URL environment variable or pass radarr_url parameter."
            )

        if not self.radarr_api_key:
            self._logger.error("Radarr API key is not configured")
            raise ValueError(
                "Radarr API key is required. Set RADARR_API_KEY environment variable or pass radarr_api_key parameter."
            )

        # Remove trailing slash from URL
        self.radarr_url = self.radarr_url.rstrip("/")

        self._logger.info(
            "RadarrClient initialized",
            radarr_url=self.radarr_url,
            component="radarr_client",
        )

    def get_all_movies(self) -> List[MovieMetadata]:
        """Fetch all movies from Radarr library.

        Returns:
            List of MovieMetadata objects for all movies in the library.

        Raises:
            ConnectionError: If Radarr is unreachable.
            ValueError: If API key is invalid or response is malformed.
            RuntimeError: If API returns an error response.
        """
        self._logger.debug("Fetching all movies from Radarr")
        return get_all_movies(self.radarr_url, self.radarr_api_key)

    def get_movie_by_tmdb_id(self, tmdb_id: int) -> Optional[MovieMetadata]:
        """Fetch a movie from Radarr by TMDB ID.

        Args:
            tmdb_id: The Movie Database (TMDB) unique identifier.

        Returns:
            MovieMetadata object if found, None otherwise.

        Raises:
            ConnectionError: If Radarr is unreachable.
            ValueError: If API key is invalid or response is malformed.
            RuntimeError: If API returns an error response.
        """
        self._logger.debug("Looking up movie by TMDB ID", tmdb_id=tmdb_id)
        return orchestrate_get_movie_by_tmdb_id(
            tmdb_id=tmdb_id,
            base_url=self.radarr_url,
            api_key=self.radarr_api_key,
        )

    def get_movie_by_radarr_id(self, radarr_id: int) -> Optional[MovieMetadata]:
        """Fetch a movie from Radarr by Radarr ID.

        Args:
            radarr_id: Radarr's internal movie identifier.

        Returns:
            MovieMetadata object if found, None otherwise.

        Raises:
            ConnectionError: If Radarr is unreachable.
            ValueError: If API key is invalid or response is malformed.
            RuntimeError: If API returns an error response.
        """
        self._logger.debug("Looking up movie by Radarr ID", radarr_id=radarr_id)
        all_movies = self.get_all_movies()
        
        matching_movies = [movie for movie in all_movies if movie.radarr_id == radarr_id]
        
        if not matching_movies:
            self._logger.info("No movie found with Radarr ID", radarr_id=radarr_id)
            return None
        
        return matching_movies[0]
