"""Pure utility for making HTTP requests to TMDB API.

Handles error handling, rate limiting, and response parsing.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

import time
from typing import Dict, Any, Optional

import requests

from src.core.logging_config import get_logger

logger = get_logger(__name__)


def make_tmdb_request(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    api_key: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """Make an HTTP GET request to TMDB API with error handling and retries.

    Args:
        url: The full URL to request.
        params: Optional query parameters (API key will be added automatically).
        api_key: TMDB API key for authentication.
        max_retries: Maximum number of retry attempts for rate limiting.
        retry_delay: Initial delay in seconds between retries (doubles each retry).

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        ValueError: If API key is missing or invalid.
        ConnectionError: If request fails after all retries.
        RuntimeError: If API returns an error response (4xx/5xx).
    """
    if not api_key:
        logger.error("TMDB API key is missing")
        raise ValueError("TMDB API key is required")

    # Prepare request parameters
    request_params = params.copy() if params else {}
    request_params["api_key"] = api_key

    logger.debug("Making TMDB API request", url=url, params=request_params)

    # Retry loop for rate limiting
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=request_params, timeout=10)

            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limited by TMDB API, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit exceeded, no retries left")
                    raise RuntimeError("TMDB API rate limit exceeded after all retries")

            # Handle 404 Not Found
            if response.status_code == 404:
                logger.warning("Resource not found", url=url, status_code=404)
                raise RuntimeError(f"TMDB resource not found: {url}")

            # Handle other client/server errors
            if response.status_code >= 400:
                logger.error(
                    "TMDB API error",
                    status_code=response.status_code,
                    response_text=response.text[:200],
                )
                raise RuntimeError(
                    f"TMDB API error {response.status_code}: {response.text[:200]}"
                )

            # Success
            logger.info("TMDB API request successful", url=url, status_code=response.status_code)
            return response.json()

        except requests.exceptions.Timeout as e:
            logger.error("TMDB API request timeout", url=url, attempt=attempt + 1)
            if attempt == max_retries - 1:
                raise ConnectionError(f"TMDB API timeout after {max_retries} attempts") from e
            time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            logger.error(
                "TMDB API request failed",
                url=url,
                error=str(e),
                attempt=attempt + 1,
            )
            if attempt == max_retries - 1:
                raise ConnectionError(f"TMDB API request failed: {str(e)}") from e
            time.sleep(retry_delay)

    # Should not reach here, but for type safety
    raise ConnectionError("TMDB API request failed after all retries")
