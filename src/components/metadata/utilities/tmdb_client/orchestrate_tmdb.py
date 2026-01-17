"""Orchestrator for TMDB operations.

Coordinates searching for movies, fetching details, and fetching credits.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Optional
from dataclasses import replace

from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.tmdb_client.api_request import make_tmdb_request
from src.components.metadata.utilities.tmdb_client.response_parser import (
    parse_movie_details,
    parse_credits,
)
from src.core.logging_config import get_logger, LoggedOperation

logger = get_logger(__name__)


def search_movie_id(
    title: str,
    *,
    year: Optional[int] = None,
    api_key: str,
    base_url: str = "https://api.themoviedb.org/3",
) -> int:
    """Search for a movie by title and optional year, returning TMDB ID.

    Args:
        title: Movie title to search for.
        year: Optional release year to narrow search results.
        api_key: TMDB API key for authentication.
        base_url: TMDB API base URL (default: official API endpoint).

    Returns:
        TMDB movie ID of the first matching result.

    Raises:
        ValueError: If title is empty or no results found.
        RuntimeError: If API request fails.
    """
    if not title or not title.strip():
        raise ValueError("Movie title cannot be empty")

    url = f"{base_url}/search/movie"
    params = {"query": title}

    if year is not None:
        params["year"] = str(year)

    logger.info("Searching for movie", title=title, year=year)

    with LoggedOperation("search_movie", logger, title=title, year=year) as op:
        response = make_tmdb_request(url, params=params, api_key=api_key)

        results = response.get("results", [])
        if not results:
            logger.warning("No results found for movie search", title=title, year=year)
            raise ValueError(f"No results found for movie: {title}")

        movie_id = results[0].get("id")
        if movie_id is None:
            raise RuntimeError("Invalid search response: missing movie ID")

        op.add_context(movie_id=movie_id, result_count=len(results))
        logger.info("Found movie", title=title, tmdb_id=movie_id)

        return movie_id


def fetch_full_metadata(
    tmdb_id: int,
    *,
    api_key: str,
    base_url: str = "https://api.themoviedb.org/3",
    top_cast: int = 20,
) -> MovieMetadata:
    """Fetch complete movie metadata including cast from TMDB.

    Args:
        tmdb_id: TMDB movie ID.
        api_key: TMDB API key for authentication.
        base_url: TMDB API base URL (default: official API endpoint).
        top_cast: Maximum number of cast members to retrieve (default: 20).

    Returns:
        Complete MovieMetadata object with movie details and cast.

    Raises:
        ValueError: If tmdb_id is invalid.
        RuntimeError: If API requests fail.
    """
    if tmdb_id <= 0:
        raise ValueError(f"Invalid TMDB ID: {tmdb_id}")

    logger.info("Fetching full metadata", tmdb_id=tmdb_id, top_cast=top_cast)

    with LoggedOperation("fetch_full_metadata", logger, tmdb_id=tmdb_id) as op:
        # Fetch movie details
        details_url = f"{base_url}/movie/{tmdb_id}"
        details_response = make_tmdb_request(details_url, api_key=api_key)
        movie_metadata = parse_movie_details(details_response)

        # Fetch credits
        credits_url = f"{base_url}/movie/{tmdb_id}/credits"
        credits_response = make_tmdb_request(credits_url, api_key=api_key)
        cast_members = parse_credits(credits_response, top_n=top_cast)

        # Merge cast into metadata (using dataclass replace since frozen)
        complete_metadata = replace(movie_metadata, cast=cast_members)

        op.add_context(
            title=complete_metadata.title,
            year=complete_metadata.year,
            cast_count=len(cast_members),
            genre_count=len(complete_metadata.genres),
        )

        logger.info(
            "Fetched complete metadata",
            tmdb_id=tmdb_id,
            title=complete_metadata.title,
            cast_count=len(cast_members),
        )

        return complete_metadata
