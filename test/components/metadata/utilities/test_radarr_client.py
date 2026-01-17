"""Tests for Radarr client utilities.

Tests the three-layer architecture: api_request, response_parser, and orchestrate_radarr.
Adheres to Memoirr testing standards: mock at import location, proper fixtures.
"""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from src.components.metadata.utilities.radarr_client.api_request import make_radarr_request
from src.components.metadata.utilities.radarr_client.response_parser import (
    parse_radarr_movie,
    extract_file_path,
)
from src.components.metadata.utilities.radarr_client.orchestrate_radarr import (
    get_all_movies,
    get_movie_by_tmdb_id,
)
from src.components.metadata.utilities.types import MovieMetadata


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_radarr_movie():
    """Sample Radarr movie JSON response."""
    return {
        "id": 1,
        "title": "The Matrix",
        "year": 1999,
        "tmdbId": 603,
        "genres": [{"name": "Action"}, {"name": "Science Fiction"}],
        "overview": "A computer hacker learns the truth about reality.",
        "movieFile": {
            "path": "/movies/The.Matrix.1999.mkv",
            "size": 1234567890,
        }
    }


@pytest.fixture
def sample_radarr_movie_no_file():
    """Sample Radarr movie without a file."""
    return {
        "id": 2,
        "title": "The Matrix Reloaded",
        "year": 2003,
        "tmdbId": 604,
        "genres": [{"name": "Action"}],
        "overview": "Neo and the rebels continue their fight.",
    }


# ============================================================================
# Tests: api_request.py
# ============================================================================

class TestMakeRadarrRequest:
    """Tests for make_radarr_request utility."""

    def test_successful_request(self):
        """Test successful API request with proper headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"title": "Test Movie"}

        with patch("src.components.metadata.utilities.radarr_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = make_radarr_request(
                url="http://localhost:7878/api/v3/movie",
                api_key="test-api-key",
            )

            assert result == {"title": "Test Movie"}
            mock_get.assert_called_once()
            # Verify X-Api-Key header
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["headers"]["X-Api-Key"] == "test-api-key"

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API key is required"):
            make_radarr_request(
                url="http://localhost:7878/api/v3/movie",
                api_key="",
            )

    def test_401_unauthorized(self):
        """Test handling of 401 Unauthorized (invalid API key)."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("src.components.metadata.utilities.radarr_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(ValueError, match="Invalid Radarr API key"):
                make_radarr_request(
                    url="http://localhost:7878/api/v3/movie",
                    api_key="bad-key",
                )

    def test_404_not_found(self):
        """Test handling of 404 Not Found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("src.components.metadata.utilities.radarr_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="resource not found"):
                make_radarr_request(
                    url="http://localhost:7878/api/v3/movie/999",
                    api_key="test-key",
                )

    def test_connection_error(self):
        """Test handling when Radarr is unreachable."""
        with patch("src.components.metadata.utilities.radarr_client.api_request.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

            with pytest.raises(ConnectionError, match="Cannot connect to Radarr"):
                make_radarr_request(
                    url="http://localhost:7878/api/v3/movie",
                    api_key="test-key",
                    max_retries=2,
                )

            # Should retry
            assert mock_get.call_count == 2

    def test_timeout_with_retry(self):
        """Test timeout handling with retries."""
        with patch("src.components.metadata.utilities.radarr_client.api_request.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Timeout")

            with pytest.raises(ConnectionError, match="timeout after"):
                make_radarr_request(
                    url="http://localhost:7878/api/v3/movie",
                    api_key="test-key",
                    max_retries=3,
                )

            assert mock_get.call_count == 3


# ============================================================================
# Tests: response_parser.py
# ============================================================================

class TestParseRadarrMovie:
    """Tests for parse_radarr_movie utility."""

    def test_parse_complete_movie(self, sample_radarr_movie):
        """Test parsing a complete movie response."""
        result = parse_radarr_movie(sample_radarr_movie)

        assert isinstance(result, MovieMetadata)
        assert result.title == "The Matrix"
        assert result.year == 1999
        assert result.tmdb_id == 603
        assert result.radarr_id == 1
        assert result.plex_rating_key is None
        assert result.cast == []  # Radarr doesn't provide cast
        assert result.genres == ["Action", "Science Fiction"]
        assert "reality" in result.overview

    def test_parse_movie_missing_title(self, sample_radarr_movie):
        """Test error when title is missing."""
        del sample_radarr_movie["title"]

        with pytest.raises(ValueError, match="missing 'title'"):
            parse_radarr_movie(sample_radarr_movie)

    def test_parse_movie_missing_year(self, sample_radarr_movie):
        """Test error when year is missing."""
        del sample_radarr_movie["year"]

        with pytest.raises(ValueError, match="missing 'year'"):
            parse_radarr_movie(sample_radarr_movie)

    def test_parse_movie_missing_tmdb_id(self, sample_radarr_movie):
        """Test error when tmdbId is missing."""
        del sample_radarr_movie["tmdbId"]

        with pytest.raises(ValueError, match="missing 'tmdbId'"):
            parse_radarr_movie(sample_radarr_movie)

    def test_parse_movie_no_genres(self, sample_radarr_movie):
        """Test parsing movie with no genres."""
        del sample_radarr_movie["genres"]

        result = parse_radarr_movie(sample_radarr_movie)
        assert result.genres == []

    def test_parse_movie_no_overview(self, sample_radarr_movie):
        """Test parsing movie with no overview."""
        del sample_radarr_movie["overview"]

        result = parse_radarr_movie(sample_radarr_movie)
        assert result.overview is None

    def test_parse_movie_genres_as_strings(self, sample_radarr_movie):
        """Test parsing movie with genres as strings (real Radarr format)."""
        # Real Radarr API returns genres as strings, not dicts
        sample_radarr_movie["genres"] = ["Action", "Science Fiction", "Adventure"]

        result = parse_radarr_movie(sample_radarr_movie)
        assert result.genres == ["Action", "Science Fiction", "Adventure"]

    def test_parse_movie_genres_mixed_format(self, sample_radarr_movie):
        """Test parsing movie with mixed genre formats."""
        # Handle both string and dict formats
        sample_radarr_movie["genres"] = [
            "Action",
            {"name": "Science Fiction"},
            "Adventure"
        ]

        result = parse_radarr_movie(sample_radarr_movie)
        assert result.genres == ["Action", "Science Fiction", "Adventure"]


class TestExtractFilePath:
    """Tests for extract_file_path utility."""

    def test_extract_existing_path(self, sample_radarr_movie):
        """Test extracting file path when present."""
        path = extract_file_path(sample_radarr_movie)
        assert path == "/movies/The.Matrix.1999.mkv"

    def test_extract_no_movie_file(self, sample_radarr_movie_no_file):
        """Test extracting when no movieFile field."""
        path = extract_file_path(sample_radarr_movie_no_file)
        assert path is None

    def test_extract_movie_file_no_path(self, sample_radarr_movie):
        """Test extracting when movieFile exists but has no path."""
        del sample_radarr_movie["movieFile"]["path"]

        path = extract_file_path(sample_radarr_movie)
        assert path is None


# ============================================================================
# Tests: orchestrate_radarr.py
# ============================================================================

class TestGetAllMovies:
    """Tests for get_all_movies orchestrator."""

    def test_successful_fetch(self, sample_radarr_movie, sample_radarr_movie_no_file):
        """Test fetching all movies successfully."""
        mock_response = [sample_radarr_movie, sample_radarr_movie_no_file]

        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.make_radarr_request") as mock_request:
            mock_request.return_value = mock_response

            movies = get_all_movies(
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            assert len(movies) == 2
            assert movies[0].title == "The Matrix"
            assert movies[1].title == "The Matrix Reloaded"
            mock_request.assert_called_once_with(
                "http://localhost:7878/api/v3/movie",
                "test-key",
            )

    def test_empty_library(self):
        """Test fetching from empty library."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.make_radarr_request") as mock_request:
            mock_request.return_value = []

            movies = get_all_movies(
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            assert movies == []

    def test_invalid_response_type(self):
        """Test error when API returns non-list response."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.make_radarr_request") as mock_request:
            mock_request.return_value = {"error": "Something went wrong"}

            with pytest.raises(ValueError, match="Expected list from Radarr API"):
                get_all_movies(
                    base_url="http://localhost:7878",
                    api_key="test-key",
                )

    def test_skip_invalid_entries(self, sample_radarr_movie):
        """Test that invalid movie entries are skipped."""
        invalid_movie = {"id": 999}  # Missing required fields
        mock_response = [sample_radarr_movie, invalid_movie]

        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.make_radarr_request") as mock_request:
            mock_request.return_value = mock_response

            movies = get_all_movies(
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            # Should skip invalid entry
            assert len(movies) == 1
            assert movies[0].title == "The Matrix"

    def test_connection_error_propagates(self):
        """Test that connection errors are propagated."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.make_radarr_request") as mock_request:
            mock_request.side_effect = ConnectionError("Cannot connect")

            with pytest.raises(ConnectionError, match="Cannot connect"):
                get_all_movies(
                    base_url="http://localhost:7878",
                    api_key="test-key",
                )


class TestGetMovieByTmdbId:
    """Tests for get_movie_by_tmdb_id orchestrator."""

    def test_movie_found(self, sample_radarr_movie):
        """Test finding movie by TMDB ID."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.get_all_movies") as mock_get_all:
            mock_movie = MovieMetadata(
                title="The Matrix",
                year=1999,
                tmdb_id=603,
                radarr_id=1,
                plex_rating_key=None,
                cast=[],
            )
            mock_get_all.return_value = [mock_movie]

            result = get_movie_by_tmdb_id(
                tmdb_id=603,
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            assert result is not None
            assert result.title == "The Matrix"
            assert result.tmdb_id == 603

    def test_movie_not_found(self):
        """Test when movie is not in library."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.get_all_movies") as mock_get_all:
            mock_movie = MovieMetadata(
                title="The Matrix",
                year=1999,
                tmdb_id=603,
                radarr_id=1,
                plex_rating_key=None,
                cast=[],
            )
            mock_get_all.return_value = [mock_movie]

            result = get_movie_by_tmdb_id(
                tmdb_id=999,  # Not in library
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            assert result is None

    def test_multiple_matches_uses_first(self):
        """Test that first match is returned when multiple movies have same TMDB ID."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.get_all_movies") as mock_get_all:
            movie1 = MovieMetadata(
                title="The Matrix",
                year=1999,
                tmdb_id=603,
                radarr_id=1,
                plex_rating_key=None,
                cast=[],
            )
            movie2 = MovieMetadata(
                title="The Matrix (Duplicate)",
                year=1999,
                tmdb_id=603,
                radarr_id=2,
                plex_rating_key=None,
                cast=[],
            )
            mock_get_all.return_value = [movie1, movie2]

            result = get_movie_by_tmdb_id(
                tmdb_id=603,
                base_url="http://localhost:7878",
                api_key="test-key",
            )

            assert result.radarr_id == 1  # First one

    def test_connection_error_propagates(self):
        """Test that connection errors are propagated."""
        with patch("src.components.metadata.utilities.radarr_client.orchestrate_radarr.get_all_movies") as mock_get_all:
            mock_get_all.side_effect = ConnectionError("Cannot connect")

            with pytest.raises(ConnectionError, match="Cannot connect"):
                get_movie_by_tmdb_id(
                    tmdb_id=603,
                    base_url="http://localhost:7878",
                    api_key="test-key",
                )
