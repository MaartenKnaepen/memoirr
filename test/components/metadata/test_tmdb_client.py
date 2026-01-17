"""Tests for TMDB client and utilities.

Tests the complete TMDB client implementation including API requests,
response parsing, and orchestration.

Adheres to Memoirr coding standards: comprehensive mocking, proper test structure.
"""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from src.components.metadata.tmdb_client import TmdbClient
from src.components.metadata.utilities.tmdb_client.api_request import make_tmdb_request
from src.components.metadata.utilities.tmdb_client.response_parser import (
    parse_movie_details,
    parse_credits,
)
from src.components.metadata.utilities.tmdb_client.orchestrate_tmdb import (
    search_movie_id,
    fetch_full_metadata,
)
from src.components.metadata.utilities.types import MovieMetadata, CastMember


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_movie_details_response():
    """Mock TMDB movie details API response."""
    return {
        "id": 603,
        "title": "The Matrix",
        "original_title": "The Matrix",
        "release_date": "1999-03-31",
        "overview": "Set in the 22nd century, The Matrix tells the story...",
        "genres": [
            {"id": 28, "name": "Action"},
            {"id": 878, "name": "Science Fiction"},
        ],
    }


@pytest.fixture
def mock_credits_response():
    """Mock TMDB credits API response."""
    return {
        "id": 603,
        "cast": [
            {
                "id": 6384,
                "name": "Keanu Reeves",
                "character": "Neo",
                "profile_path": "/path/to/keanu.jpg",
            },
            {
                "id": 2975,
                "name": "Laurence Fishburne",
                "character": "Morpheus",
                "profile_path": "/path/to/laurence.jpg",
            },
            {
                "id": 530,
                "name": "Carrie-Anne Moss",
                "character": "Trinity",
                "profile_path": None,
            },
        ],
    }


@pytest.fixture
def mock_search_response():
    """Mock TMDB search API response."""
    return {
        "page": 1,
        "results": [
            {
                "id": 603,
                "title": "The Matrix",
                "release_date": "1999-03-31",
                "overview": "Set in the 22nd century...",
            }
        ],
        "total_pages": 1,
        "total_results": 1,
    }


# ============================================================================
# Tests for api_request.py
# ============================================================================


class TestMakeTmdbRequest:
    """Tests for make_tmdb_request utility function."""

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_successful_request(self, mock_get):
        """Test successful API request returns JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 603, "title": "The Matrix"}
        mock_get.return_value = mock_response

        result = make_tmdb_request(
            "https://api.themoviedb.org/3/movie/603",
            api_key="test_key",
        )

        assert result == {"id": 603, "title": "The Matrix"}
        mock_get.assert_called_once()

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_api_key_added_to_params(self, mock_get):
        """Test that API key is added to request parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        make_tmdb_request(
            "https://api.themoviedb.org/3/movie/603",
            params={"language": "en-US"},
            api_key="test_key",
        )

        # Verify API key was added to params
        call_args = mock_get.call_args
        assert call_args[1]["params"]["api_key"] == "test_key"
        assert call_args[1]["params"]["language"] == "en-US"

    def test_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/603",
                api_key="",
            )

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_404_error(self, mock_get):
        """Test that 404 error raises RuntimeError."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="not found"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/999999",
                api_key="test_key",
            )

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_500_error(self, mock_get):
        """Test that 500 error raises RuntimeError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="API error 500"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/603",
                api_key="test_key",
            )

    @patch("src.components.metadata.utilities.tmdb_client.api_request.time.sleep")
    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_rate_limit_retry_success(self, mock_get, mock_sleep):
        """Test that 429 rate limit error triggers retry and eventually succeeds."""
        # First call returns 429, second call succeeds
        mock_response_429 = Mock()
        mock_response_429.status_code = 429

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"id": 603}

        mock_get.side_effect = [mock_response_429, mock_response_200]

        result = make_tmdb_request(
            "https://api.themoviedb.org/3/movie/603",
            api_key="test_key",
            max_retries=3,
        )

        assert result == {"id": 603}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("src.components.metadata.utilities.tmdb_client.api_request.time.sleep")
    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_rate_limit_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test that exceeding max retries raises RuntimeError."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/603",
                api_key="test_key",
                max_retries=2,
            )

        assert mock_get.call_count == 2

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_timeout_error(self, mock_get):
        """Test that timeout raises ConnectionError."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")

        with pytest.raises(ConnectionError, match="timeout"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/603",
                api_key="test_key",
                max_retries=1,
            )

    @patch("src.components.metadata.utilities.tmdb_client.api_request.requests.get")
    def test_connection_error(self, mock_get):
        """Test that connection error raises ConnectionError."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(ConnectionError, match="request failed"):
            make_tmdb_request(
                "https://api.themoviedb.org/3/movie/603",
                api_key="test_key",
                max_retries=1,
            )


# ============================================================================
# Tests for response_parser.py
# ============================================================================


class TestParseMovieDetails:
    """Tests for parse_movie_details utility function."""

    def test_parse_complete_response(self, mock_movie_details_response):
        """Test parsing complete movie details response."""
        result = parse_movie_details(mock_movie_details_response)

        assert isinstance(result, MovieMetadata)
        assert result.title == "The Matrix"
        assert result.year == 1999
        assert result.tmdb_id == 603
        assert result.overview == "Set in the 22nd century, The Matrix tells the story..."
        assert result.genres == ["Action", "Science Fiction"]
        assert result.cast == []  # Cast populated separately
        assert result.radarr_id is None
        assert result.plex_rating_key is None

    def test_parse_missing_title(self):
        """Test that missing title raises ValueError."""
        response = {"id": 603, "release_date": "1999-03-31"}

        with pytest.raises(ValueError, match="Missing required field: title"):
            parse_movie_details(response)

    def test_parse_missing_id(self):
        """Test that missing ID raises ValueError."""
        response = {"title": "The Matrix", "release_date": "1999-03-31"}

        with pytest.raises(ValueError, match="Missing required field: id"):
            parse_movie_details(response)

    def test_parse_invalid_release_date(self):
        """Test that invalid release date defaults to year 0."""
        response = {
            "id": 603,
            "title": "The Matrix",
            "release_date": "invalid-date",
        }

        result = parse_movie_details(response)
        assert result.year == 0

    def test_parse_missing_release_date(self):
        """Test that missing release date defaults to year 0."""
        response = {"id": 603, "title": "The Matrix"}

        result = parse_movie_details(response)
        assert result.year == 0

    def test_parse_empty_genres(self):
        """Test parsing response with no genres."""
        response = {
            "id": 603,
            "title": "The Matrix",
            "release_date": "1999-03-31",
            "genres": [],
        }

        result = parse_movie_details(response)
        assert result.genres == []

    def test_parse_missing_overview(self):
        """Test parsing response without overview."""
        response = {
            "id": 603,
            "title": "The Matrix",
            "release_date": "1999-03-31",
        }

        result = parse_movie_details(response)
        assert result.overview is None


class TestParseCredits:
    """Tests for parse_credits utility function."""

    def test_parse_complete_credits(self, mock_credits_response):
        """Test parsing complete credits response."""
        result = parse_credits(mock_credits_response, top_n=20)

        assert len(result) == 3
        assert all(isinstance(member, CastMember) for member in result)

        # Check first cast member
        assert result[0].name == "Keanu Reeves"
        assert result[0].character == "Neo"
        assert result[0].tmdb_id == 6384
        assert result[0].profile_path == "https://image.tmdb.org/t/p/w185/path/to/keanu.jpg"

        # Check cast member without profile path
        assert result[2].name == "Carrie-Anne Moss"
        assert result[2].profile_path is None

    def test_parse_credits_top_n_limit(self, mock_credits_response):
        """Test that top_n parameter limits returned cast members."""
        # Add more cast members to test limiting
        mock_credits_response["cast"].extend(
            [
                {"id": i, "name": f"Actor {i}", "character": f"Character {i}"}
                for i in range(4, 25)
            ]
        )

        result = parse_credits(mock_credits_response, top_n=5)
        assert len(result) == 5

    def test_parse_empty_cast(self):
        """Test parsing credits with empty cast list."""
        response = {"id": 603, "cast": []}

        result = parse_credits(response)
        assert result == []

    def test_parse_missing_cast_field(self):
        """Test parsing response without cast field."""
        response = {"id": 603}

        result = parse_credits(response)
        assert result == []

    def test_parse_invalid_cast_data_structure(self):
        """Test that invalid cast data structure raises ValueError."""
        response = {"cast": "not a list"}

        with pytest.raises(ValueError, match="expected list"):
            parse_credits(response)

    def test_skip_cast_member_with_missing_fields(self):
        """Test that cast members with missing essential fields are skipped."""
        response = {
            "cast": [
                {"id": 1, "name": "Actor 1", "character": "Character 1"},
                {"id": 2, "name": "Actor 2"},  # Missing character
                {"name": "Actor 3", "character": "Character 3"},  # Missing id
                {"id": 4, "character": "Character 4"},  # Missing name
                {"id": 5, "name": "Actor 5", "character": "Character 5"},
            ]
        }

        result = parse_credits(response)
        assert len(result) == 2  # Only first and last are valid


# ============================================================================
# Tests for orchestrate_tmdb.py
# ============================================================================


class TestSearchMovieId:
    """Tests for search_movie_id orchestrator function."""

    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.LoggedOperation")
    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.make_tmdb_request")
    def test_search_success(self, mock_request, mock_logged_op, mock_search_response):
        """Test successful movie search returns ID."""
        # Setup LoggedOperation mock
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time()
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance

        mock_request.return_value = mock_search_response

        result = search_movie_id("The Matrix", api_key="test_key")

        assert result == 603
        mock_request.assert_called_once()
        mock_op_instance.add_context.assert_called_once()

    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.LoggedOperation")
    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.make_tmdb_request")
    def test_search_with_year(self, mock_request, mock_logged_op, mock_search_response):
        """Test movie search with year parameter."""
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time()
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance

        mock_request.return_value = mock_search_response

        result = search_movie_id("The Matrix", year=1999, api_key="test_key")

        assert result == 603
        # Verify year was included in parameters
        call_args = mock_request.call_args
        assert call_args[1]["params"]["year"] == "1999"

    def test_search_empty_title(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            search_movie_id("", api_key="test_key")

    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.LoggedOperation")
    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.make_tmdb_request")
    def test_search_no_results(self, mock_request, mock_logged_op):
        """Test that no search results raises ValueError."""
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time()
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance

        mock_request.return_value = {"results": []}

        with pytest.raises(ValueError, match="No results found"):
            search_movie_id("NonexistentMovie12345", api_key="test_key")


class TestFetchFullMetadata:
    """Tests for fetch_full_metadata orchestrator function."""

    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.LoggedOperation")
    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.make_tmdb_request")
    def test_fetch_complete_metadata(
        self, mock_request, mock_logged_op, mock_movie_details_response, mock_credits_response
    ):
        """Test fetching complete metadata with details and credits."""
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time()
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance

        # Return details first, then credits
        mock_request.side_effect = [mock_movie_details_response, mock_credits_response]

        result = fetch_full_metadata(603, api_key="test_key")

        assert isinstance(result, MovieMetadata)
        assert result.title == "The Matrix"
        assert result.tmdb_id == 603
        assert len(result.cast) == 3
        assert result.cast[0].name == "Keanu Reeves"
        assert mock_request.call_count == 2

    def test_fetch_invalid_id(self):
        """Test that invalid TMDB ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TMDB ID"):
            fetch_full_metadata(0, api_key="test_key")

        with pytest.raises(ValueError, match="Invalid TMDB ID"):
            fetch_full_metadata(-1, api_key="test_key")

    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.LoggedOperation")
    @patch("src.components.metadata.utilities.tmdb_client.orchestrate_tmdb.make_tmdb_request")
    def test_fetch_with_top_cast_limit(
        self, mock_request, mock_logged_op, mock_movie_details_response, mock_credits_response
    ):
        """Test that top_cast parameter is passed to parse_credits."""
        mock_op_instance = Mock()
        mock_op_instance.start_time = time.time()
        mock_logged_op.return_value.__enter__.return_value = mock_op_instance

        mock_request.side_effect = [mock_movie_details_response, mock_credits_response]

        result = fetch_full_metadata(603, api_key="test_key", top_cast=2)

        # Should only have 2 cast members even though response has 3
        assert len(result.cast) == 2


# ============================================================================
# Tests for TmdbClient wrapper
# ============================================================================


class TestTmdbClient:
    """Tests for TmdbClient wrapper class."""

    @patch("src.components.metadata.tmdb_client.get_settings")
    def test_init_success(self, mock_settings):
        """Test successful initialization with valid API key."""
        mock_settings.return_value = Mock(
            tmdb_api_key="test_key",
            tmdb_base_url="https://api.themoviedb.org/3",
        )

        client = TmdbClient()

        assert client.api_key == "test_key"
        assert client.base_url == "https://api.themoviedb.org/3"

    @patch("src.components.metadata.tmdb_client.get_settings")
    def test_init_missing_api_key(self, mock_settings):
        """Test that missing API key raises ValueError."""
        mock_settings.return_value = Mock(
            tmdb_api_key=None,
            tmdb_base_url="https://api.themoviedb.org/3",
        )

        with pytest.raises(ValueError, match="TMDB_API_KEY is required"):
            TmdbClient()

    @patch("src.components.metadata.tmdb_client.search_movie_id")
    @patch("src.components.metadata.tmdb_client.get_settings")
    def test_search_movie(self, mock_settings, mock_search):
        """Test search_movie method."""
        mock_settings.return_value = Mock(
            tmdb_api_key="test_key",
            tmdb_base_url="https://api.themoviedb.org/3",
        )
        mock_search.return_value = 603

        client = TmdbClient()
        result = client.search_movie("The Matrix", year=1999)

        assert result == 603
        mock_search.assert_called_once_with(
            "The Matrix",
            year=1999,
            api_key="test_key",
            base_url="https://api.themoviedb.org/3",
        )

    @patch("src.components.metadata.tmdb_client.fetch_full_metadata")
    @patch("src.components.metadata.tmdb_client.get_settings")
    def test_get_movie_metadata_by_id(self, mock_settings, mock_fetch):
        """Test get_movie_metadata_by_id method."""
        mock_settings.return_value = Mock(
            tmdb_api_key="test_key",
            tmdb_base_url="https://api.themoviedb.org/3",
        )
        mock_metadata = MovieMetadata(
            title="The Matrix",
            year=1999,
            tmdb_id=603,
            radarr_id=None,
            plex_rating_key=None,
            cast=[],
            genres=["Action"],
        )
        mock_fetch.return_value = mock_metadata

        client = TmdbClient()
        result = client.get_movie_metadata_by_id(603, top_cast=15)

        assert result == mock_metadata
        mock_fetch.assert_called_once_with(
            603,
            api_key="test_key",
            base_url="https://api.themoviedb.org/3",
            top_cast=15,
        )

    @patch("src.components.metadata.tmdb_client.search_movie_id")
    @patch("src.components.metadata.tmdb_client.fetch_full_metadata")
    @patch("src.components.metadata.tmdb_client.get_settings")
    def test_get_movie_metadata(self, mock_settings, mock_fetch, mock_search):
        """Test get_movie_metadata convenience method."""
        mock_settings.return_value = Mock(
            tmdb_api_key="test_key",
            tmdb_base_url="https://api.themoviedb.org/3",
        )
        mock_search.return_value = 603
        mock_metadata = MovieMetadata(
            title="The Matrix",
            year=1999,
            tmdb_id=603,
            radarr_id=None,
            plex_rating_key=None,
            cast=[],
            genres=["Action"],
        )
        mock_fetch.return_value = mock_metadata

        client = TmdbClient()
        result = client.get_movie_metadata("The Matrix", year=1999, top_cast=10)

        assert result == mock_metadata
        mock_search.assert_called_once()
        mock_fetch.assert_called_once()
