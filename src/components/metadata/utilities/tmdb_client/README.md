# TMDB Client

Pure utilities for fetching movie metadata from The Movie Database (TMDB) API.

## Overview

The TMDB client follows the 3-layer Memoirr architecture:

1. **Wrapper Layer** (`tmdb_client.py`): Manages configuration and provides high-level interface
2. **Orchestrator Layer** (`orchestrate_tmdb.py`): Coordinates API operations
3. **Utility Layer** (`api_request.py`, `response_parser.py`): Pure functions for HTTP and parsing

## Usage

### Basic Usage

```python
from src.components.metadata import TmdbClient

# Initialize client (requires TMDB_API_KEY in environment)
client = TmdbClient()

# Search for a movie and fetch complete metadata
metadata = client.get_movie_metadata("The Matrix", year=1999)

print(f"Title: {metadata.title}")
print(f"Year: {metadata.year}")
print(f"TMDB ID: {metadata.tmdb_id}")
print(f"Cast: {len(metadata.cast)} members")
for cast_member in metadata.cast[:3]:
    print(f"  - {cast_member.name} as {cast_member.character}")
```

### Advanced Usage

```python
# Search for movie ID only
tmdb_id = client.search_movie("Inception", year=2010)

# Fetch metadata by ID with custom cast limit
metadata = client.get_movie_metadata_by_id(tmdb_id, top_cast=10)
```

## Configuration

Set the following environment variables (or add to `.env`):

```bash
# Required
TMDB_API_KEY=your_tmdb_api_key_here

# Optional (defaults shown)
TMDB_BASE_URL=https://api.themoviedb.org/3
```

Get your API key at: https://www.themoviedb.org/settings/api

## Features

- **Error Handling**: Comprehensive error handling for API failures (404, 500, timeout, etc.)
- **Rate Limiting**: Automatic retry with exponential backoff for 429 rate limit errors
- **Type Safety**: Returns frozen dataclasses (`MovieMetadata`, `CastMember`)
- **Structured Logging**: Full logging with `structlog` for debugging and monitoring
- **Configurable**: Cast member limit, base URL, and retry parameters

## API Methods

### `TmdbClient`

#### `__init__() -> None`
Initialize client with configuration from settings. Raises `ValueError` if API key is missing.

#### `search_movie(title: str, *, year: Optional[int] = None) -> int`
Search for a movie and return its TMDB ID.

**Args:**
- `title`: Movie title to search for
- `year`: Optional release year to narrow results

**Returns:** TMDB movie ID

**Raises:**
- `ValueError`: If title is empty or no results found
- `RuntimeError`: If API request fails

#### `get_movie_metadata(title: str, *, year: Optional[int] = None, top_cast: int = 20) -> MovieMetadata`
Search for a movie and fetch complete metadata (convenience method).

**Args:**
- `title`: Movie title to search for
- `year`: Optional release year to narrow results
- `top_cast`: Maximum cast members to retrieve (default: 20)

**Returns:** Complete `MovieMetadata` object with cast

#### `get_movie_metadata_by_id(tmdb_id: int, *, top_cast: int = 20) -> MovieMetadata`
Fetch complete movie metadata using TMDB ID.

**Args:**
- `tmdb_id`: TMDB movie ID
- `top_cast`: Maximum cast members to retrieve (default: 20)

**Returns:** Complete `MovieMetadata` object with cast

## Data Structures

### `MovieMetadata`
```python
@dataclass(frozen=True)
class MovieMetadata:
    title: str
    year: int
    tmdb_id: int
    radarr_id: Optional[int]
    plex_rating_key: Optional[str]
    cast: List[CastMember]
    genres: List[str]
    overview: Optional[str]
```

### `CastMember`
```python
@dataclass(frozen=True)
class CastMember:
    name: str
    character: str
    tmdb_id: int
    profile_path: Optional[str]  # Full URL to headshot image
```

## Testing

Run the comprehensive test suite:

```bash
uv run pytest test/components/metadata/test_tmdb_client.py -v
```

34 tests covering:
- API request handling (success, errors, retries, rate limiting)
- Response parsing (complete/partial data, missing fields)
- Orchestration (search, metadata fetching)
- Client wrapper (initialization, methods)

## Architecture Details

### Layer 1: Utilities

**`api_request.py`**
- Pure function for HTTP requests: `make_tmdb_request()`
- Handles rate limiting (429) with exponential backoff
- Handles errors (404, 500) with clear exceptions
- Timeout and connection error retry logic

**`response_parser.py`**
- Pure functions for parsing JSON: `parse_movie_details()`, `parse_credits()`
- Transforms raw TMDB responses into frozen dataclasses
- Validates required fields, provides sensible defaults

### Layer 2: Orchestrator

**`orchestrate_tmdb.py`**
- Coordinates API calls: `search_movie_id()`, `fetch_full_metadata()`
- Combines multiple endpoints (details + credits)
- Uses `LoggedOperation` for timing and metrics

### Layer 3: Wrapper

**`tmdb_client.py`**
- Manages API key configuration from `Settings`
- Provides clean, high-level interface
- Validates configuration on initialization

## Next Steps

This client provides the foundation for:
1. **Phase 2**: Radarr and Plex clients (similar architecture)
2. **Phase 3**: Face recognition pipeline (uses cast metadata)
3. **Phase 4**: Speaker tagging (uses character names)
