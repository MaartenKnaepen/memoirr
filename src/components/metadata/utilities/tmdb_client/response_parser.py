"""Pure functions to parse TMDB API responses into dataclasses.

Transforms raw TMDB JSON into MovieMetadata and CastMember frozen dataclasses.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Dict, Any, List, Optional

from src.components.metadata.utilities.types import MovieMetadata, CastMember
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def parse_movie_details(data: Dict[str, Any]) -> MovieMetadata:
    """Parse TMDB movie details response into MovieMetadata.

    Args:
        data: Raw JSON response from TMDB /movie/{id} endpoint.

    Returns:
        MovieMetadata object with basic movie information (without cast).

    Raises:
        ValueError: If required fields are missing from the response.
    """
    try:
        # Extract required fields
        title = data.get("title") or data.get("original_title")
        if not title:
            raise ValueError("Missing required field: title")

        tmdb_id = data.get("id")
        if tmdb_id is None:
            raise ValueError("Missing required field: id")

        # Extract year from release_date (format: "YYYY-MM-DD")
        release_date = data.get("release_date", "")
        try:
            year = int(release_date[:4]) if release_date else 0
        except (ValueError, IndexError):
            logger.warning("Invalid release_date format", release_date=release_date)
            year = 0

        # Extract genres
        genres = [genre["name"] for genre in data.get("genres", []) if "name" in genre]

        # Extract overview
        overview = data.get("overview")

        logger.debug(
            "Parsed movie details",
            title=title,
            year=year,
            tmdb_id=tmdb_id,
            genre_count=len(genres),
        )

        return MovieMetadata(
            title=title,
            year=year,
            tmdb_id=tmdb_id,
            radarr_id=None,
            plex_rating_key=None,
            cast=[],  # Will be populated by parse_credits
            genres=genres,
            overview=overview,
        )

    except KeyError as e:
        logger.error("Missing required field in TMDB response", field=str(e))
        raise ValueError(f"Invalid TMDB movie details response: missing {str(e)}") from e


def parse_credits(data: Dict[str, Any], *, top_n: int = 20) -> List[CastMember]:
    """Parse TMDB credits response into list of CastMember objects.

    Args:
        data: Raw JSON response from TMDB /movie/{id}/credits endpoint.
        top_n: Maximum number of cast members to extract (default: 20).

    Returns:
        List of CastMember objects, ordered by appearance (top billing first).

    Raises:
        ValueError: If the response structure is invalid.
    """
    try:
        cast_list = data.get("cast", [])

        if not isinstance(cast_list, list):
            raise ValueError("Invalid cast data: expected list")

        cast_members: List[CastMember] = []

        # Process cast members (already ordered by billing)
        for cast_data in cast_list[:top_n]:
            try:
                name = cast_data.get("name")
                character = cast_data.get("character")
                cast_id = cast_data.get("id")

                # Skip if essential fields are missing
                if not name or not character or cast_id is None:
                    logger.warning(
                        "Skipping cast member with missing fields",
                        name=name,
                        character=character,
                        id=cast_id,
                    )
                    continue

                profile_path = cast_data.get("profile_path")
                # Convert relative path to full URL if present
                if profile_path:
                    profile_path = f"https://image.tmdb.org/t/p/w185{profile_path}"

                cast_members.append(
                    CastMember(
                        name=name,
                        character=character,
                        tmdb_id=cast_id,
                        profile_path=profile_path,
                    )
                )

            except Exception as e:
                logger.warning(
                    "Failed to parse cast member",
                    cast_data=cast_data,
                    error=str(e),
                )
                continue

        logger.debug("Parsed credits", cast_count=len(cast_members), top_n=top_n)

        return cast_members

    except Exception as e:
        logger.error("Failed to parse credits response", error=str(e))
        raise ValueError(f"Invalid TMDB credits response: {str(e)}") from e
