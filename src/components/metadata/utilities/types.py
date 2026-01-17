"""Data structures for movie metadata.

This module defines frozen dataclasses for movie metadata, including cast information
and movie details from TMDB, Radarr, and Plex.

Adheres to Memoirr coding standards: frozen dataclasses, type hints, Google-style docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CastMember:
    """Represents a cast member in a movie.
    
    Attributes:
        name: Actor's full name.
        character: Character name portrayed by the actor.
        tmdb_id: The Movie Database (TMDB) unique identifier for the actor.
        profile_path: Optional URL or path to the actor's headshot/profile image.
    """
    name: str
    character: str
    tmdb_id: int
    profile_path: Optional[str] = None


@dataclass(frozen=True)
class MovieMetadata:
    """Represents comprehensive metadata for a movie.
    
    Attributes:
        title: Movie title.
        year: Release year.
        tmdb_id: The Movie Database (TMDB) unique identifier.
        radarr_id: Optional Radarr application identifier for movie management.
        plex_rating_key: Optional Plex Media Server rating key for library integration.
        cast: List of cast members appearing in the movie.
        genres: List of genre tags (e.g., "Action", "Drama", "Fantasy").
        overview: Optional plot synopsis or description.
    """
    title: str
    year: int
    tmdb_id: int
    radarr_id: Optional[int]
    plex_rating_key: Optional[str]
    cast: List[CastMember]
    genres: List[str] = field(default_factory=list)
    overview: Optional[str] = None
