"""Parse Plex JSON responses.

Plex API responses are notoriously nested within a MediaContainer structure.
This module provides utilities to extract relevant data from these responses.

Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from typing import Dict, Any, Optional, List

from src.components.metadata.utilities.types import MovieMetadata, CastMember
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def extract_server_identity(data: Dict[str, Any]) -> str:
    """Extract server machine identifier from Plex identity endpoint response.

    Args:
        data: JSON response from /identity endpoint.

    Returns:
        The machineIdentifier (server ID) needed for deep links.

    Raises:
        ValueError: If machineIdentifier is not found in response.
    """
    # Plex wraps identity response in MediaContainer
    media_container = data.get("MediaContainer", {})
    machine_id = media_container.get("machineIdentifier")
    
    # Fallback: check at top level (older Plex versions)
    if not machine_id:
        machine_id = data.get("machineIdentifier")
    
    if not machine_id:
        logger.error("machineIdentifier not found in Plex identity response", data=data)
        raise ValueError("Plex identity response missing 'machineIdentifier' field")
    
    logger.debug("Extracted Plex server identity", machine_identifier=machine_id)
    return machine_id


def parse_plex_metadata(data: Dict[str, Any]) -> MovieMetadata:
    """Parse Plex metadata response into MovieMetadata dataclass.

    Plex API returns data nested within MediaContainer > Metadata array.
    This function extracts the first movie result and parses it.

    Args:
        data: JSON response from Plex search or metadata endpoints.

    Returns:
        MovieMetadata object with parsed information.

    Raises:
        ValueError: If response structure is invalid or no results found.
    """
    # Plex wraps results in MediaContainer
    media_container = data.get("MediaContainer", {})
    
    # Get metadata array (list of results)
    metadata_list = media_container.get("Metadata", [])
    
    if not metadata_list:
        logger.warning("No metadata found in Plex response")
        raise ValueError("No movies found in Plex response")
    
    # Take the first result
    movie_data = metadata_list[0]
    
    # Extract basic fields
    title = movie_data.get("title", "Unknown")
    year = movie_data.get("year", 0)
    rating_key = movie_data.get("ratingKey")
    
    if not rating_key:
        logger.error("ratingKey not found in Plex movie data", movie_data=movie_data)
        raise ValueError("Plex movie data missing 'ratingKey' field")
    
    # Try to extract TMDB ID from guid field
    # Plex guid format examples: "plex://movie/5d776825880197001ec967c1", "tmdb://603"
    tmdb_id = _extract_tmdb_id_from_guid(movie_data.get("Guid", []))
    
    # Parse cast if available
    cast = _parse_cast(movie_data.get("Role", []))
    
    # Parse genres if available
    genres = _parse_genres(movie_data.get("Genre", []))
    
    # Extract overview/summary
    overview = movie_data.get("summary")
    
    logger.info(
        "Parsed Plex metadata",
        title=title,
        year=year,
        rating_key=rating_key,
        tmdb_id=tmdb_id,
        cast_count=len(cast),
    )
    
    return MovieMetadata(
        title=title,
        year=year,
        tmdb_id=tmdb_id if tmdb_id else 0,  # Default to 0 if not found
        radarr_id=None,
        plex_rating_key=rating_key,
        cast=cast,
        genres=genres,
        overview=overview,
    )


def _extract_tmdb_id_from_guid(guid_list: List[Dict[str, Any]]) -> Optional[int]:
    """Extract TMDB ID from Plex Guid array.

    Plex stores multiple GUID sources. We look for tmdb:// prefix.

    Args:
        guid_list: List of guid objects from Plex metadata.

    Returns:
        TMDB ID as integer if found, None otherwise.
    """
    for guid_obj in guid_list:
        guid_str = guid_obj.get("id", "")
        if guid_str.startswith("tmdb://"):
            try:
                tmdb_id = int(guid_str.replace("tmdb://", ""))
                logger.debug("Extracted TMDB ID from Plex guid", tmdb_id=tmdb_id)
                return tmdb_id
            except ValueError:
                logger.warning("Invalid TMDB ID format in Plex guid", guid=guid_str)
    
    logger.debug("No TMDB ID found in Plex guid list")
    return None


def _parse_cast(role_list: List[Dict[str, Any]]) -> List[CastMember]:
    """Parse Plex Role array into CastMember objects.

    Args:
        role_list: List of role objects from Plex metadata.

    Returns:
        List of CastMember dataclasses.
    """
    cast = []
    for role in role_list:
        name = role.get("tag", "Unknown")
        character = role.get("role", "Unknown")
        # Plex doesn't always provide TMDB IDs for actors, use a placeholder
        tmdb_id = role.get("id", 0)
        profile_path = role.get("thumb")
        
        cast.append(CastMember(
            name=name,
            character=character,
            tmdb_id=tmdb_id,
            profile_path=profile_path,
        ))
    
    return cast


def _parse_genres(genre_list: List[Dict[str, Any]]) -> List[str]:
    """Parse Plex Genre array into list of genre strings.

    Args:
        genre_list: List of genre objects from Plex metadata.

    Returns:
        List of genre names.
    """
    return [genre.get("tag", "") for genre in genre_list if genre.get("tag")]


def extract_file_path(movie_data: Dict[str, Any]) -> Optional[str]:
    """Extract the absolute file path from a Plex movie metadata object.

    Plex stores file paths in a nested Media > Part structure.
    This function extracts the first file path found.

    Args:
        movie_data: A single movie metadata dict from Plex Metadata array.

    Returns:
        Absolute path to the video file, or None if no file is present.
    """
    media_list = movie_data.get("Media", [])
    if not media_list:
        logger.debug("No Media array in Plex response", rating_key=movie_data.get("ratingKey"))
        return None

    # Get first media entry
    media = media_list[0]
    parts = media.get("Part", [])
    if not parts:
        logger.debug("No Part array in Plex Media", rating_key=movie_data.get("ratingKey"))
        return None

    # Get file path from first part
    file_path = parts[0].get("file")
    if file_path:
        logger.debug("Extracted file path from Plex", path=file_path)
    else:
        logger.warning("Part present but no file field", rating_key=movie_data.get("ratingKey"))

    return file_path


def extract_file_path_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract file path from a full Plex API response.

    This is a convenience function that handles the MediaContainer wrapper.

    Args:
        data: Full JSON response from Plex metadata endpoint.

    Returns:
        Absolute path to the video file, or None if not found.
    """
    media_container = data.get("MediaContainer", {})
    metadata_list = media_container.get("Metadata", [])

    if not metadata_list:
        logger.debug("No Metadata in Plex response for file path extraction")
        return None

    return extract_file_path(metadata_list[0])
