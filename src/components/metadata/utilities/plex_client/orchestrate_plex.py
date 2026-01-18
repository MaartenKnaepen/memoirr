"""Orchestrator for Plex client workflows.

Coordinates API requests and response parsing for Plex Media Server operations.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from src.components.metadata.utilities.types import MovieMetadata
from src.components.metadata.utilities.plex_client.api_request import make_plex_request
from src.components.metadata.utilities.plex_client.response_parser import (
    parse_plex_metadata,
    extract_server_identity,
    extract_file_path,
)
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def get_server_identity(base_url: str, token: str) -> str:
    """Retrieve Plex server machine identifier.

    Args:
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.

    Returns:
        Server machine identifier (needed for deep links).

    Raises:
        ValueError: If authentication fails or response is invalid.
        ConnectionError: If Plex server is unreachable.
    """
    identity_url = f"{base_url.rstrip('/')}/identity"
    
    logger.info("Fetching Plex server identity", url=identity_url)
    
    try:
        response_data = make_plex_request(identity_url, token)
        server_id = extract_server_identity(response_data)
        
        logger.info("Successfully retrieved Plex server identity", server_id=server_id)
        return server_id
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to get Plex server identity", error=str(e))
        raise


def search_plex_library(
    title: str,
    base_url: str,
    token: str,
    *,
    year: Optional[int] = None,
) -> Optional[MovieMetadata]:
    """Search Plex library for a movie by title.

    Args:
        title: Movie title to search for.
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.
        year: Optional release year to refine search.

    Returns:
        MovieMetadata object if found, None if no results.

    Raises:
        ValueError: If authentication fails or response is invalid.
        ConnectionError: If Plex server is unreachable.
    """
    search_url = f"{base_url.rstrip('/')}/hubs/search"
    
    params = {
        "query": title,
        "limit": 5,
        "includeCollections": 0,
    }
    
    if year:
        params["year"] = year
    
    logger.info("Searching Plex library", title=title, year=year)
    
    try:
        response_data = make_plex_request(search_url, token, params=params)
        
        # Parse the first movie result
        metadata = parse_plex_metadata(response_data)
        
        logger.info(
            "Found movie in Plex library",
            title=metadata.title,
            rating_key=metadata.plex_rating_key,
        )
        return metadata
        
    except ValueError as e:
        # No results found
        logger.warning("Movie not found in Plex library", title=title, error=str(e))
        return None
        
    except (ConnectionError, RuntimeError) as e:
        logger.error("Failed to search Plex library", error=str(e))
        raise


def generate_deep_link(
    server_id: str,
    rating_key: str,
    timestamp_ms: int,
    *,
    client_type: str = "web",
) -> str:
    """Generate a Plex deep link URL for playing a specific scene.

    Args:
        server_id: Plex server machine identifier.
        rating_key: Plex media rating key (unique identifier for the movie).
        timestamp_ms: Timestamp in milliseconds to start playback.
        client_type: Target client type ("web" or "desktop").

    Returns:
        Deep link URL string.

    Raises:
        ValueError: If client_type is invalid.
    """
    if client_type not in ("web", "desktop"):
        logger.error("Invalid client type", client_type=client_type)
        raise ValueError(f"Invalid client_type: {client_type}. Must be 'web' or 'desktop'.")
    
    if client_type == "web":
        # Web client uses app.plex.tv with URL-encoded key parameter
        # Note: quote() encodes slashes by default, safe='' ensures complete encoding
        encoded_key = quote(f"/library/metadata/{rating_key}", safe='')
        deep_link = (
            f"https://app.plex.tv/desktop/#!/server/{server_id}/details"
            f"?key={encoded_key}&t={timestamp_ms}"
        )
    else:  # desktop
        # Desktop client uses plex:// protocol
        deep_link = f"plex://server/{server_id}/library/metadata/{rating_key}?t={timestamp_ms}"
    
    logger.debug(
        "Generated Plex deep link",
        server_id=server_id,
        rating_key=rating_key,
        timestamp_ms=timestamp_ms,
        client_type=client_type,
    )
    
    return deep_link


def get_movie_by_rating_key(
    rating_key: str,
    base_url: str,
    token: str,
) -> MovieMetadata:
    """Retrieve detailed movie metadata by rating key.

    Args:
        rating_key: Plex media rating key.
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.

    Returns:
        MovieMetadata object with full details.

    Raises:
        ValueError: If authentication fails or movie not found.
        ConnectionError: If Plex server is unreachable.
    """
    metadata_url = f"{base_url.rstrip('/')}/library/metadata/{rating_key}"
    
    logger.info("Fetching Plex movie metadata", rating_key=rating_key)
    
    try:
        response_data = make_plex_request(metadata_url, token)
        metadata = parse_plex_metadata(response_data)
        
        logger.info("Successfully retrieved movie metadata", title=metadata.title)
        return metadata
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to get movie metadata", rating_key=rating_key, error=str(e))
        raise


def get_movie_library_sections(
    base_url: str,
    token: str,
) -> List[Dict[str, str]]:
    """Get all movie library sections from Plex.

    Args:
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.

    Returns:
        List of dicts with 'key' and 'title' for each movie library section.

    Raises:
        ValueError: If authentication fails.
        ConnectionError: If Plex server is unreachable.
    """
    sections_url = f"{base_url.rstrip('/')}/library/sections"
    
    logger.info("Fetching Plex library sections")
    
    try:
        response_data = make_plex_request(sections_url, token)
        media_container = response_data.get("MediaContainer", {})
        directories = media_container.get("Directory", [])
        
        # Filter for movie libraries only
        movie_sections = [
            {"key": section.get("key"), "title": section.get("title")}
            for section in directories
            if section.get("type") == "movie"
        ]
        
        logger.info("Found movie library sections", count=len(movie_sections))
        return movie_sections
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to get library sections", error=str(e))
        raise


def get_all_movies_with_paths(
    base_url: str,
    token: str,
    *,
    section_key: Optional[str] = None,
) -> List[Tuple[str, str, Optional[str]]]:
    """Get all movies from Plex library with their file paths.

    Args:
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.
        section_key: Optional library section key. If None, searches all movie sections.

    Returns:
        List of tuples: (rating_key, title, file_path).
        file_path may be None if no file is associated.

    Raises:
        ValueError: If authentication fails.
        ConnectionError: If Plex server is unreachable.
    """
    logger.info("Fetching all movies with file paths from Plex")
    
    results: List[Tuple[str, str, Optional[str]]] = []
    
    try:
        # Determine which sections to query
        if section_key:
            section_keys = [section_key]
        else:
            sections = get_movie_library_sections(base_url, token)
            section_keys = [s["key"] for s in sections if s["key"]]
        
        for key in section_keys:
            # Use includeGuids to get external IDs, and includeMedia to get file info
            all_url = f"{base_url.rstrip('/')}/library/sections/{key}/all"
            params = {
                "includeGuids": 1,
            }
            
            response_data = make_plex_request(all_url, token, params=params)
            media_container = response_data.get("MediaContainer", {})
            metadata_list = media_container.get("Metadata", [])
            
            for movie_data in metadata_list:
                rating_key = movie_data.get("ratingKey")
                title = movie_data.get("title", "Unknown")
                file_path = extract_file_path(movie_data)
                
                if rating_key:
                    results.append((rating_key, title, file_path))
        
        logger.info("Retrieved movies with paths from Plex", count=len(results))
        return results
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to get all movies with paths", error=str(e))
        raise


def find_movie_by_file_path(
    file_path: str,
    base_url: str,
    token: str,
    *,
    section_key: Optional[str] = None,
) -> Optional[str]:
    """Find a Plex rating key by matching a file path.

    This function searches the Plex library for a movie whose file path
    matches the provided path. Useful for correlating Radarr file paths
    with Plex rating keys.

    Args:
        file_path: Absolute file path to search for (from Radarr).
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.
        section_key: Optional library section key to narrow search.

    Returns:
        Plex rating key if found, None otherwise.

    Raises:
        ValueError: If authentication fails.
        ConnectionError: If Plex server is unreachable.
    """
    logger.info("Searching Plex for movie by file path", file_path=file_path)
    
    # Normalize the input path for comparison
    normalized_input = Path(file_path).resolve()
    
    try:
        all_movies = get_all_movies_with_paths(base_url, token, section_key=section_key)
        
        for rating_key, title, plex_path in all_movies:
            if plex_path:
                # Normalize Plex path for comparison
                normalized_plex = Path(plex_path).resolve()
                
                if normalized_input == normalized_plex:
                    logger.info(
                        "Found matching movie in Plex",
                        title=title,
                        rating_key=rating_key,
                        file_path=file_path,
                    )
                    return rating_key
        
        logger.warning("No matching movie found in Plex for file path", file_path=file_path)
        return None
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to find movie by file path", file_path=file_path, error=str(e))
        raise


def build_file_path_to_rating_key_map(
    base_url: str,
    token: str,
    *,
    section_key: Optional[str] = None,
) -> Dict[str, str]:
    """Build a mapping from file paths to Plex rating keys.

    This function creates a dictionary that can be used for fast lookups
    when correlating multiple Radarr file paths with Plex rating keys.

    Args:
        base_url: Plex server base URL (e.g., "http://localhost:32400").
        token: Plex authentication token.
        section_key: Optional library section key to narrow search.

    Returns:
        Dict mapping normalized file paths to rating keys.

    Raises:
        ValueError: If authentication fails.
        ConnectionError: If Plex server is unreachable.
    """
    logger.info("Building file path to rating key map from Plex")
    
    path_map: Dict[str, str] = {}
    
    try:
        all_movies = get_all_movies_with_paths(base_url, token, section_key=section_key)
        
        for rating_key, title, file_path in all_movies:
            if file_path:
                # Normalize path for consistent lookups
                normalized_path = str(Path(file_path).resolve())
                path_map[normalized_path] = rating_key
        
        logger.info("Built file path to rating key map", entry_count=len(path_map))
        return path_map
        
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to build file path map", error=str(e))
        raise
