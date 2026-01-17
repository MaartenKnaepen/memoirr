"""Orchestrator for Radarr API workflows.

Coordinates API requests and response parsing for movie metadata retrieval.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import List, Optional

from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.radarr_client.api_request import make_radarr_request
from src.components.metadata.utilities.radarr_client.response_parser import parse_radarr_movie
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def get_all_movies(
    base_url: str,
    api_key: str,
) -> List[MovieMetadata]:
    """Fetch all movies from Radarr library.

    Args:
        base_url: Radarr server base URL (e.g., "http://localhost:7878").
        api_key: Radarr API key for authentication.

    Returns:
        List of MovieMetadata objects for all movies in the library.

    Raises:
        ConnectionError: If Radarr is unreachable.
        ValueError: If API key is invalid or response is malformed.
        RuntimeError: If API returns an error response.
    """
    url = f"{base_url}/api/v3/movie"
    
    logger.info("Fetching all movies from Radarr", url=url)
    
    try:
        response = make_radarr_request(url, api_key)
        
        # Response should be a list of movie objects
        if not isinstance(response, list):
            logger.error("Unexpected response type from Radarr /movie endpoint", type=type(response).__name__)
            raise ValueError(f"Expected list from Radarr API, got {type(response).__name__}")
        
        movies = []
        for movie_data in response:
            try:
                movie = parse_radarr_movie(movie_data)
                movies.append(movie)
            except ValueError as e:
                # Log and skip invalid movie entries
                logger.warning(
                    "Skipping invalid movie entry",
                    error=str(e),
                    movie_id=movie_data.get("id"),
                )
                continue
        
        logger.info("Successfully fetched movies from Radarr", count=len(movies))
        return movies
        
    except (ConnectionError, ValueError, RuntimeError) as e:
        logger.error("Failed to fetch movies from Radarr", error=str(e))
        raise


def get_movie_by_tmdb_id(
    tmdb_id: int,
    base_url: str,
    api_key: str,
) -> Optional[MovieMetadata]:
    """Fetch a movie from Radarr by TMDB ID.

    Radarr doesn't have a direct "get by TMDB ID" endpoint, so we fetch all movies
    and filter client-side. For production use with large libraries, consider using
    the /api/v3/movie/lookup endpoint or caching the movie list.

    Args:
        tmdb_id: The Movie Database (TMDB) unique identifier.
        base_url: Radarr server base URL (e.g., "http://localhost:7878").
        api_key: Radarr API key for authentication.

    Returns:
        MovieMetadata object if found, None otherwise.

    Raises:
        ConnectionError: If Radarr is unreachable.
        ValueError: If API key is invalid or response is malformed.
        RuntimeError: If API returns an error response.
    """
    logger.info("Looking up movie by TMDB ID in Radarr", tmdb_id=tmdb_id)
    
    try:
        all_movies = get_all_movies(base_url, api_key)
        
        # Filter for matching TMDB ID
        matching_movies = [movie for movie in all_movies if movie.tmdb_id == tmdb_id]
        
        if not matching_movies:
            logger.info("No movie found with TMDB ID in Radarr", tmdb_id=tmdb_id)
            return None
        
        if len(matching_movies) > 1:
            logger.warning(
                "Multiple movies found with same TMDB ID (using first)",
                tmdb_id=tmdb_id,
                count=len(matching_movies),
            )
        
        movie = matching_movies[0]
        logger.info(
            "Found movie in Radarr",
            tmdb_id=tmdb_id,
            title=movie.title,
            year=movie.year,
        )
        return movie
        
    except (ConnectionError, ValueError, RuntimeError) as e:
        logger.error("Failed to lookup movie by TMDB ID", tmdb_id=tmdb_id, error=str(e))
        raise
