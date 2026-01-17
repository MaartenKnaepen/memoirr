"""Parse Radarr JSON responses into MovieMetadata objects.

Radarr returns "Movie" objects with embedded "MovieFile" objects.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Dict, Any, Optional

from src.components.metadata.utilities.types import MovieMetadata
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def parse_radarr_movie(data: Dict[str, Any]) -> MovieMetadata:
    """Parse a Radarr movie JSON response into MovieMetadata.

    Args:
        data: Raw JSON dictionary from Radarr API movie endpoint.

    Returns:
        MovieMetadata object with parsed data.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    try:
        title = data.get("title")
        year = data.get("year")
        tmdb_id = data.get("tmdbId")
        radarr_id = data.get("id")

        # Validate required fields
        if not title:
            logger.error("Missing 'title' in Radarr response", data_sample=str(data)[:200])
            raise ValueError("Radarr response missing 'title' field")

        if not year:
            logger.error("Missing 'year' in Radarr response", data_sample=str(data)[:200])
            raise ValueError("Radarr response missing 'year' field")

        if not tmdb_id:
            logger.error("Missing 'tmdbId' in Radarr response", data_sample=str(data)[:200])
            raise ValueError("Radarr response missing 'tmdbId' field")

        # Extract optional fields
        # Radarr can return genres as strings or as dicts with "name" key
        raw_genres = data.get("genres", [])
        genres = []
        for genre in raw_genres:
            if isinstance(genre, str):
                genres.append(genre)
            elif isinstance(genre, dict):
                genres.append(genre.get("name", ""))
            else:
                logger.warning("Unexpected genre type", genre_type=type(genre).__name__)
        
        overview = data.get("overview")

        # Note: Radarr doesn't provide cast information in the movie endpoint
        # Cast would need to be fetched separately from TMDB
        logger.debug(
            "Parsed Radarr movie",
            title=title,
            year=year,
            tmdb_id=tmdb_id,
            radarr_id=radarr_id,
        )

        return MovieMetadata(
            title=title,
            year=year,
            tmdb_id=tmdb_id,
            radarr_id=radarr_id,
            plex_rating_key=None,  # Radarr doesn't provide Plex info
            cast=[],  # Radarr doesn't provide cast info
            genres=genres,
            overview=overview,
        )

    except KeyError as e:
        logger.error("Missing expected field in Radarr response", field=str(e))
        raise ValueError(f"Invalid Radarr movie data structure: missing {str(e)}") from e


def extract_file_path(movie_data: Dict[str, Any]) -> Optional[str]:
    """Extract the absolute file path from a Radarr movie object.

    Args:
        movie_data: Raw JSON dictionary from Radarr API movie endpoint.

    Returns:
        Absolute path to the video file, or None if no file is present.
    """
    movie_file = movie_data.get("movieFile")
    if not movie_file:
        logger.debug("No movieFile present in Radarr response", movie_id=movie_data.get("id"))
        return None

    path = movie_file.get("path")
    if path:
        logger.debug("Extracted file path from Radarr", path=path)
    else:
        logger.warning("movieFile present but no path field", movie_file_sample=str(movie_file)[:100])

    return path
