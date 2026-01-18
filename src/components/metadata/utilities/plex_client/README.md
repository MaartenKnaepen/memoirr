# Plex Client Utilities

This package provides utilities for interacting with Plex Media Server API to retrieve movie metadata and generate deep links for playback.

## Architecture

Follows the 3-layer Memoirr architecture:

1. **api_request.py**: Pure utility for HTTP requests with authentication
2. **response_parser.py**: Parse Plex's nested JSON responses
3. **orchestrate_plex.py**: Coordinate API calls and parsing

## Key Features

### Server Identity
Retrieve the machine identifier needed for deep links:
```python
from src.components.metadata.utilities.plex_client.orchestrate_plex import get_server_identity

server_id = get_server_identity("http://localhost:32400", "your-token")
```

### Search Library
Find movies in your Plex library:
```python
from src.components.metadata.utilities.plex_client.orchestrate_plex import search_plex_library

metadata = search_plex_library(
    title="The Matrix",
    base_url="http://localhost:32400",
    token="your-token",
    year=1999  # Optional
)
```

### Generate Deep Links
Create universal deep links for web and desktop clients:
```python
from src.components.metadata.utilities.plex_client.orchestrate_plex import generate_deep_link

# Web client link
web_link = generate_deep_link(
    server_id="abc123",
    rating_key="1234",
    timestamp_ms=300000,  # 5 minutes
    client_type="web"
)

# Desktop client link (plex:// protocol)
desktop_link = generate_deep_link(
    server_id="abc123",
    rating_key="1234",
    timestamp_ms=300000,
    client_type="desktop"
)
```

### File Path Correlation (Radarr Integration)
Match file paths from Radarr with Plex rating keys:
```python
from src.components.metadata.utilities.plex_client.orchestrate_plex import (
    find_movie_by_file_path,
    build_file_path_to_rating_key_map,
    get_all_movies_with_paths,
)

# Single file lookup
rating_key = find_movie_by_file_path(
    file_path="/movies/The Matrix (1999)/The Matrix (1999).mkv",
    base_url="http://localhost:32400",
    token="your-token"
)

# Build a map for batch lookups (more efficient for multiple files)
path_map = build_file_path_to_rating_key_map(
    base_url="http://localhost:32400",
    token="your-token"
)
# Then lookup: rating_key = path_map.get(normalized_path)

# Get all movies with their paths
movies = get_all_movies_with_paths(
    base_url="http://localhost:32400",
    token="your-token"
)
# Returns: [(rating_key, title, file_path), ...]
```

## Deep Link Formats

### Web Client
```
https://app.plex.tv/desktop/#!/server/{server_id}/details?key=%2Flibrary%2Fmetadata%2F{rating_key}&t={timestamp_ms}
```

### Desktop Client
```
plex://server/{server_id}/library/metadata/{rating_key}?t={timestamp_ms}
```

## Error Handling

All functions raise appropriate exceptions:
- `ValueError`: Invalid parameters or authentication failures
- `ConnectionError`: Network issues or server unreachable
- `RuntimeError`: API errors (4xx/5xx responses)

## Configuration

Requires environment variables:
- `PLEX_URL`: Your Plex server URL (e.g., "http://localhost:32400")
- `PLEX_TOKEN`: Your Plex authentication token

Get your token from: https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/
