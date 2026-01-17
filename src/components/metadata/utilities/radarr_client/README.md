# Radarr Client Utilities

This module provides utilities for interacting with the Radarr API to resolve local video files to movie metadata.

## Architecture

Follows the three-layer pattern:

1. **Component** (`radarr_client.py`): Wrapper class with config loading
2. **Orchestrator** (`orchestrate_radarr.py`): Coordinates API calls and parsing
3. **Utilities** (`api_request.py`, `response_parser.py`): Pure functions

## Usage

```python
from src.components.metadata.radarr_client import RadarrClient

# Initialize with config from environment
client = RadarrClient()

# Or override with explicit values
client = RadarrClient(
    radarr_url="http://localhost:7878",
    radarr_api_key="your-api-key"
)

# Fetch all movies
movies = client.get_all_movies()

# Look up by TMDB ID
movie = client.get_movie_by_tmdb_id(tmdb_id=123)
```

## Configuration

Required environment variables:
- `RADARR_URL`: Radarr server URL (e.g., "http://localhost:7878")
- `RADARR_API_KEY`: Radarr API key (found in Settings > General)

## API Reference

### get_all_movies()
Fetches all movies from the Radarr library.

### get_movie_by_tmdb_id(tmdb_id: int)
Looks up a movie by its TMDB ID. Returns `None` if not found.

### get_movie_by_radarr_id(radarr_id: int)
Looks up a movie by its Radarr internal ID. Returns `None` if not found.

## Notes

- Radarr doesn't provide cast information. Use `TMDBClient` to enrich metadata with cast details.
- The `get_movie_by_tmdb_id` method fetches all movies and filters client-side. For large libraries, consider caching.
- File paths are available via the `movieFile` field in Radarr responses. Use `extract_file_path()` from `response_parser.py`.
