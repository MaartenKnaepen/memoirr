"""Plex Media Server client component.

This component provides methods to interact with Plex Media Server, including
searching for movies, retrieving metadata, and generating deep links for playback.

Note: This is NOT a Haystack component (no @component decorator) as it serves
as a wrapper/client for Plex API operations used by other components.

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Optional

from src.core.config import get_settings
from src.core.logging_config import get_logger, MetricsLogger
from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.plex_client.orchestrate_plex import (
    get_server_identity,
    search_plex_library,
    generate_deep_link,
    get_movie_by_rating_key,
    find_movie_by_file_path,
    build_file_path_to_rating_key_map,
    get_all_movies_with_paths,
)


class PlexClient:
    """Client for interacting with Plex Media Server.

    This class provides a high-level interface for Plex operations including
    movie search, metadata retrieval, and deep link generation.

    Attributes:
        base_url: Plex server base URL.
        token: Plex authentication token.
        server_id: Cached Plex server machine identifier (fetched on init if reachable).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        """Initialize Plex client.

        Args:
            base_url: Plex server URL. If None, reads from settings (PLEX_URL).
            token: Plex authentication token. If None, reads from settings (PLEX_TOKEN).

        Raises:
            ValueError: If base_url or token are not provided and not in settings.
        """
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)

        # Use provided values or fall back to settings
        self.base_url = base_url if base_url is not None else settings.plex_url
        self.token = token if token is not None else settings.plex_token

        if not self.base_url:
            self._logger.error("Plex URL not configured")
            raise ValueError("Plex URL must be provided or set in PLEX_URL environment variable")

        if not self.token:
            self._logger.error("Plex token not configured")
            raise ValueError("Plex token must be provided or set in PLEX_TOKEN environment variable")

        # Try to fetch and cache server ID on initialization
        self.server_id: Optional[str] = None
        try:
            self.server_id = get_server_identity(self.base_url, self.token)
            self._logger.info(
                "PlexClient initialized with cached server ID",
                base_url=self.base_url,
                server_id=self.server_id,
            )
            self._metrics.counter("plex_client_init_success", 1)
        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.warning(
                "Could not fetch Plex server ID on init (will retry on demand)",
                error=str(e),
            )
            self._metrics.counter("plex_client_init_server_id_failed", 1)

    def search(
        self,
        title: str,
        *,
        year: Optional[int] = None,
    ) -> Optional[MovieMetadata]:
        """Search Plex library for a movie.

        Args:
            title: Movie title to search for.
            year: Optional release year to refine search.

        Returns:
            MovieMetadata object if found, None otherwise.

        Raises:
            ValueError: If authentication fails.
            ConnectionError: If Plex server is unreachable.
        """
        self._logger.info("Searching Plex library", title=title, year=year)
        self._metrics.counter("plex_search_requests", 1)

        try:
            result = search_plex_library(
                title=title,
                base_url=self.base_url,
                token=self.token,
                year=year,
            )

            if result:
                self._metrics.counter("plex_search_hits", 1)
            else:
                self._metrics.counter("plex_search_misses", 1)

            return result

        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.error("Plex search failed", title=title, error=str(e))
            self._metrics.counter("plex_search_errors", 1)
            raise

    def get_metadata(self, rating_key: str) -> MovieMetadata:
        """Retrieve movie metadata by rating key.

        Args:
            rating_key: Plex media rating key.

        Returns:
            MovieMetadata object with full details.

        Raises:
            ValueError: If movie not found or authentication fails.
            ConnectionError: If Plex server is unreachable.
        """
        self._logger.info("Fetching Plex metadata", rating_key=rating_key)
        self._metrics.counter("plex_metadata_requests", 1)

        try:
            result = get_movie_by_rating_key(
                rating_key=rating_key,
                base_url=self.base_url,
                token=self.token,
            )
            self._metrics.counter("plex_metadata_success", 1)
            return result

        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.error("Failed to fetch metadata", rating_key=rating_key, error=str(e))
            self._metrics.counter("plex_metadata_errors", 1)
            raise

    def get_deep_link(
        self,
        rating_key: str,
        start_ms: int,
        *,
        client_type: str = "web",
    ) -> str:
        """Generate a deep link for playing a movie at a specific timestamp.

        Args:
            rating_key: Plex media rating key.
            start_ms: Start time in milliseconds.
            client_type: Target client type ("web" or "desktop").

        Returns:
            Deep link URL string.

        Raises:
            ValueError: If server_id not available or client_type invalid.
            ConnectionError: If unable to fetch server_id.
        """
        # Ensure we have server_id (fetch if not cached)
        if not self.server_id:
            self._logger.info("Server ID not cached, fetching now")
            try:
                self.server_id = get_server_identity(self.base_url, self.token)
            except (ValueError, ConnectionError, RuntimeError) as e:
                self._logger.error("Failed to fetch server ID for deep link", error=str(e))
                self._metrics.counter("plex_deep_link_server_id_errors", 1)
                raise

        self._logger.info(
            "Generating Plex deep link",
            rating_key=rating_key,
            start_ms=start_ms,
            client_type=client_type,
        )
        self._metrics.counter("plex_deep_links_generated", 1, client_type=client_type)

        return generate_deep_link(
            server_id=self.server_id,
            rating_key=rating_key,
            timestamp_ms=start_ms,
            client_type=client_type,
        )

    def get_server_id(self) -> str:
        """Get the Plex server machine identifier.

        Returns:
            Server machine identifier.

        Raises:
            ConnectionError: If unable to fetch server_id.
        """
        if not self.server_id:
            self.server_id = get_server_identity(self.base_url, self.token)

        return self.server_id

    def find_by_file_path(self, file_path: str) -> Optional[str]:
        """Find a Plex rating key by matching a file path.

        This method searches the Plex library for a movie whose file path
        matches the provided path. Useful for correlating Radarr file paths
        with Plex rating keys.

        Args:
            file_path: Absolute file path to search for (from Radarr).

        Returns:
            Plex rating key if found, None otherwise.

        Raises:
            ValueError: If authentication fails.
            ConnectionError: If Plex server is unreachable.
        """
        self._logger.info("Finding Plex movie by file path", file_path=file_path)
        self._metrics.counter("plex_file_path_lookups", 1)

        try:
            result = find_movie_by_file_path(
                file_path=file_path,
                base_url=self.base_url,
                token=self.token,
            )

            if result:
                self._metrics.counter("plex_file_path_hits", 1)
            else:
                self._metrics.counter("plex_file_path_misses", 1)

            return result

        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.error("Failed to find movie by file path", file_path=file_path, error=str(e))
            self._metrics.counter("plex_file_path_errors", 1)
            raise

    def build_path_map(self) -> dict:
        """Build a mapping from file paths to Plex rating keys.

        This method creates a dictionary that can be used for fast lookups
        when correlating multiple Radarr file paths with Plex rating keys.
        Useful for batch operations.

        Returns:
            Dict mapping normalized file paths to rating keys.

        Raises:
            ValueError: If authentication fails.
            ConnectionError: If Plex server is unreachable.
        """
        self._logger.info("Building file path to rating key map")
        self._metrics.counter("plex_path_map_builds", 1)

        try:
            path_map = build_file_path_to_rating_key_map(
                base_url=self.base_url,
                token=self.token,
            )
            self._metrics.counter("plex_path_map_entries", len(path_map))
            return path_map

        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.error("Failed to build path map", error=str(e))
            self._metrics.counter("plex_path_map_errors", 1)
            raise

    def get_all_movies(self) -> list:
        """Get all movies from Plex library with their file paths.

        Returns:
            List of tuples: (rating_key, title, file_path).
            file_path may be None if no file is associated.

        Raises:
            ValueError: If authentication fails.
            ConnectionError: If Plex server is unreachable.
        """
        self._logger.info("Fetching all movies from Plex")
        self._metrics.counter("plex_get_all_movies", 1)

        try:
            movies = get_all_movies_with_paths(
                base_url=self.base_url,
                token=self.token,
            )
            self._metrics.counter("plex_movies_retrieved", len(movies))
            return movies

        except (ValueError, ConnectionError, RuntimeError) as e:
            self._logger.error("Failed to get all movies", error=str(e))
            self._metrics.counter("plex_get_all_movies_errors", 1)
            raise
