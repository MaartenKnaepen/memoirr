"""Pure utility for making HTTP requests to Radarr API.

Handles error handling, authentication, and response parsing.
Adheres to Memoirr coding standards: type hints, Google-style docstrings, SRP.
"""

import time
from typing import Dict, Any, List, Union, Optional

import requests

from src.core.logging_config import get_logger

logger = get_logger(__name__)


def make_radarr_request(
    url: str,
    api_key: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Make an HTTP GET request to Radarr API with error handling and retries.

    Args:
        url: The full URL to request.
        api_key: Radarr API key for authentication.
        params: Optional query parameters.
        max_retries: Maximum number of retry attempts for connection issues.
        retry_delay: Initial delay in seconds between retries (doubles each retry).

    Returns:
        Parsed JSON response as a dictionary or list of dictionaries.

    Raises:
        ValueError: If API key is missing or invalid.
        ConnectionError: If request fails after all retries or Radarr is unreachable.
        RuntimeError: If API returns an error response (4xx/5xx).
    """
    if not api_key:
        logger.error("Radarr API key is missing")
        raise ValueError("Radarr API key is required")

    # Prepare headers with API key
    headers = {"X-Api-Key": api_key}

    logger.debug("Making Radarr API request", url=url, params=params)

    # Retry loop for connection issues
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            # Handle 401 Unauthorized
            if response.status_code == 401:
                logger.error("Radarr API authentication failed", status_code=401)
                raise ValueError("Invalid Radarr API key (401 Unauthorized)")

            # Handle 404 Not Found
            if response.status_code == 404:
                logger.warning("Radarr resource not found", url=url, status_code=404)
                raise RuntimeError(f"Radarr resource not found: {url}")

            # Handle other client/server errors
            if response.status_code >= 400:
                logger.error(
                    "Radarr API error",
                    status_code=response.status_code,
                    response_text=response.text[:200],
                )
                raise RuntimeError(
                    f"Radarr API error {response.status_code}: {response.text[:200]}"
                )

            # Success
            logger.info("Radarr API request successful", url=url, status_code=response.status_code)
            return response.json()

        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Radarr connection failed (server may be down/unreachable)",
                url=url,
                attempt=attempt + 1,
                error=str(e),
            )
            if attempt == max_retries - 1:
                raise ConnectionError(
                    f"Cannot connect to Radarr at {url}. Is Radarr running?"
                ) from e
            time.sleep(retry_delay * (2 ** attempt))

        except requests.exceptions.Timeout as e:
            logger.error("Radarr API request timeout", url=url, attempt=attempt + 1)
            if attempt == max_retries - 1:
                raise ConnectionError(f"Radarr API timeout after {max_retries} attempts") from e
            time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            logger.error(
                "Radarr API request failed",
                url=url,
                error=str(e),
                attempt=attempt + 1,
            )
            if attempt == max_retries - 1:
                raise ConnectionError(f"Radarr API request failed: {str(e)}") from e
            time.sleep(retry_delay)

    # Should not reach here, but for type safety
    raise ConnectionError("Radarr API request failed after all retries")
