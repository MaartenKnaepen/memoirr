"""Tests for Plex client utilities.

Tests the utility functions for API requests, response parsing, and orchestration.
Adheres to Memoirr testing standards: mock at import location, comprehensive coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.components.metadata.utilities.plex_client.api_request import make_plex_request
from src.components.metadata.utilities.plex_client.response_parser import (
    extract_server_identity,
    parse_plex_metadata,
    _extract_tmdb_id_from_guid,
    _parse_cast,
    _parse_genres,
    extract_file_path,
    extract_file_path_from_response,
)
from src.components.metadata.utilities.plex_client.orchestrate_plex import (
    get_server_identity,
    search_plex_library,
    generate_deep_link,
    get_movie_by_rating_key,
    get_movie_library_sections,
    get_all_movies_with_paths,
    find_movie_by_file_path,
    build_file_path_to_rating_key_map,
)
from src.components.metadata.utilities.types import CastMember


class TestMakePlexRequest:
    """Tests for make_plex_request function."""

    def test_successful_request(self):
        """Test successful API request returns parsed JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"MediaContainer": {"size": 1}}

        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = make_plex_request("http://localhost:32400/identity", "test-token")

            assert result == {"MediaContainer": {"size": 1}}
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]["headers"]["X-Plex-Token"] == "test-token"
            assert call_args[1]["headers"]["Accept"] == "application/json"

    def test_missing_token_raises_error(self):
        """Test that missing token raises ValueError."""
        with pytest.raises(ValueError, match="Plex authentication token is required"):
            make_plex_request("http://localhost:32400/identity", "")

    def test_401_unauthorized_raises_error(self):
        """Test that 401 response raises ValueError for invalid token."""
        mock_response = Mock()
        mock_response.status_code = 401

        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(ValueError, match="Invalid Plex authentication token"):
                make_plex_request("http://localhost:32400/identity", "invalid-token")

    def test_404_not_found_raises_error(self):
        """Test that 404 response raises RuntimeError."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="Plex resource not found"):
                make_plex_request("http://localhost:32400/missing", "test-token")

    def test_server_error_raises_runtime_error(self):
        """Test that 5xx response raises RuntimeError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(RuntimeError, match="Plex API error 500"):
                make_plex_request("http://localhost:32400/identity", "test-token")

    def test_connection_error_with_retries(self):
        """Test that connection errors are retried and eventually raise ConnectionError."""
        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

            with pytest.raises(ConnectionError, match="Cannot connect to Plex"):
                make_plex_request("http://localhost:32400/identity", "test-token", max_retries=2)

            assert mock_get.call_count == 2

    def test_timeout_error_with_retries(self):
        """Test that timeout errors are retried and eventually raise ConnectionError."""
        with patch("src.components.metadata.utilities.plex_client.api_request.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

            with pytest.raises(ConnectionError, match="Plex API timeout"):
                make_plex_request("http://localhost:32400/identity", "test-token", max_retries=2)

            assert mock_get.call_count == 2


class TestExtractServerIdentity:
    """Tests for extract_server_identity function."""

    def test_extract_valid_machine_identifier_in_media_container(self):
        """Test extracting machine identifier from MediaContainer (current Plex format)."""
        data = {"MediaContainer": {"machineIdentifier": "abc123def456"}}
        result = extract_server_identity(data)
        assert result == "abc123def456"

    def test_extract_valid_machine_identifier_top_level(self):
        """Test extracting machine identifier from top level (legacy format)."""
        data = {"machineIdentifier": "abc123def456"}
        result = extract_server_identity(data)
        assert result == "abc123def456"

    def test_missing_machine_identifier_raises_error(self):
        """Test that missing machineIdentifier raises ValueError."""
        data = {"foo": "bar"}
        with pytest.raises(ValueError, match="missing 'machineIdentifier' field"):
            extract_server_identity(data)


class TestParseGenres:
    """Tests for _parse_genres helper function."""

    def test_parse_valid_genres(self):
        """Test parsing genre list from Plex format."""
        genre_list = [
            {"tag": "Action"},
            {"tag": "Adventure"},
            {"tag": "Sci-Fi"},
        ]
        result = _parse_genres(genre_list)
        assert result == ["Action", "Adventure", "Sci-Fi"]

    def test_parse_empty_genres(self):
        """Test parsing empty genre list."""
        result = _parse_genres([])
        assert result == []

    def test_skip_genres_without_tag(self):
        """Test that genres without 'tag' field are skipped."""
        genre_list = [
            {"tag": "Action"},
            {"id": 123},  # No tag
            {"tag": "Adventure"},
        ]
        result = _parse_genres(genre_list)
        assert result == ["Action", "Adventure"]


class TestParseCast:
    """Tests for _parse_cast helper function."""

    def test_parse_valid_cast(self):
        """Test parsing cast list from Plex format."""
        role_list = [
            {"tag": "Keanu Reeves", "role": "Neo", "id": 6384, "thumb": "/path/to/image.jpg"},
            {"tag": "Laurence Fishburne", "role": "Morpheus", "id": 2975},
        ]
        result = _parse_cast(role_list)

        assert len(result) == 2
        assert isinstance(result[0], CastMember)
        assert result[0].name == "Keanu Reeves"
        assert result[0].character == "Neo"
        assert result[0].tmdb_id == 6384
        assert result[0].profile_path == "/path/to/image.jpg"
        assert result[1].name == "Laurence Fishburne"
        assert result[1].character == "Morpheus"

    def test_parse_empty_cast(self):
        """Test parsing empty cast list."""
        result = _parse_cast([])
        assert result == []


class TestExtractTmdbId:
    """Tests for _extract_tmdb_id_from_guid helper function."""

    def test_extract_tmdb_id_from_guid(self):
        """Test extracting TMDB ID from guid list."""
        guid_list = [
            {"id": "plex://movie/5d776825880197001ec967c1"},
            {"id": "tmdb://603"},
            {"id": "imdb://tt0133093"},
        ]
        result = _extract_tmdb_id_from_guid(guid_list)
        assert result == 603

    def test_no_tmdb_id_returns_none(self):
        """Test that missing TMDB ID returns None."""
        guid_list = [
            {"id": "plex://movie/5d776825880197001ec967c1"},
            {"id": "imdb://tt0133093"},
        ]
        result = _extract_tmdb_id_from_guid(guid_list)
        assert result is None

    def test_empty_guid_list_returns_none(self):
        """Test that empty guid list returns None."""
        result = _extract_tmdb_id_from_guid([])
        assert result is None

    def test_invalid_tmdb_format_returns_none(self):
        """Test that invalid TMDB format returns None."""
        guid_list = [{"id": "tmdb://invalid"}]
        result = _extract_tmdb_id_from_guid(guid_list)
        assert result is None


class TestParsePlexMetadata:
    """Tests for parse_plex_metadata function."""

    def test_parse_complete_metadata(self):
        """Test parsing complete Plex metadata response."""
        data = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "title": "The Matrix",
                        "year": 1999,
                        "ratingKey": "12345",
                        "summary": "A computer hacker learns about the true nature of reality.",
                        "Guid": [
                            {"id": "tmdb://603"},
                        ],
                        "Role": [
                            {"tag": "Keanu Reeves", "role": "Neo", "id": 6384},
                        ],
                        "Genre": [
                            {"tag": "Action"},
                            {"tag": "Sci-Fi"},
                        ],
                    }
                ]
            }
        }

        result = parse_plex_metadata(data)

        assert result.title == "The Matrix"
        assert result.year == 1999
        assert result.plex_rating_key == "12345"
        assert result.tmdb_id == 603
        assert result.overview == "A computer hacker learns about the true nature of reality."
        assert len(result.cast) == 1
        assert result.cast[0].name == "Keanu Reeves"
        assert len(result.genres) == 2
        assert "Action" in result.genres

    def test_parse_metadata_no_results_raises_error(self):
        """Test that empty metadata list raises ValueError."""
        data = {
            "MediaContainer": {
                "Metadata": []
            }
        }

        with pytest.raises(ValueError, match="No movies found in Plex response"):
            parse_plex_metadata(data)

    def test_parse_metadata_missing_rating_key_raises_error(self):
        """Test that missing ratingKey raises ValueError."""
        data = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "title": "The Matrix",
                        "year": 1999,
                        # Missing ratingKey
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing 'ratingKey' field"):
            parse_plex_metadata(data)

    def test_parse_metadata_without_tmdb_id(self):
        """Test parsing metadata when TMDB ID is not available."""
        data = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "title": "Unknown Movie",
                        "year": 2020,
                        "ratingKey": "99999",
                        "Guid": [],
                        "Role": [],
                        "Genre": [],
                    }
                ]
            }
        }

        result = parse_plex_metadata(data)
        assert result.tmdb_id == 0  # Default value


class TestGenerateDeepLink:
    """Tests for generate_deep_link function."""

    def test_generate_web_client_link(self):
        """Test generating web client deep link."""
        link = generate_deep_link(
            server_id="abc123",
            rating_key="12345",
            timestamp_ms=300000,
            client_type="web",
        )

        assert "https://app.plex.tv/desktop/" in link
        assert "server/abc123" in link
        assert "12345" in link
        assert "t=300000" in link
        assert "%2Flibrary%2Fmetadata%2F" in link  # URL encoded

    def test_generate_desktop_client_link(self):
        """Test generating desktop client deep link."""
        link = generate_deep_link(
            server_id="abc123",
            rating_key="12345",
            timestamp_ms=300000,
            client_type="desktop",
        )

        assert link.startswith("plex://")
        assert "server/abc123" in link
        assert "library/metadata/12345" in link
        assert "t=300000" in link

    def test_invalid_client_type_raises_error(self):
        """Test that invalid client_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid client_type: invalid"):
            generate_deep_link(
                server_id="abc123",
                rating_key="12345",
                timestamp_ms=300000,
                client_type="invalid",
            )


class TestGetServerIdentity:
    """Tests for get_server_identity orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_server_identity_success(self, mock_request):
        """Test successful server identity retrieval."""
        mock_request.return_value = {"MediaContainer": {"machineIdentifier": "test-server-id"}}

        result = get_server_identity("http://localhost:32400", "test-token")

        assert result == "test-server-id"
        mock_request.assert_called_once_with("http://localhost:32400/identity", "test-token")

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_server_identity_connection_error(self, mock_request):
        """Test that connection errors are propagated."""
        mock_request.side_effect = ConnectionError("Cannot connect")

        with pytest.raises(ConnectionError):
            get_server_identity("http://localhost:32400", "test-token")


class TestSearchPlexLibrary:
    """Tests for search_plex_library orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_search_library_success(self, mock_request):
        """Test successful library search."""
        mock_request.return_value = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "title": "The Matrix",
                        "year": 1999,
                        "ratingKey": "12345",
                        "Guid": [{"id": "tmdb://603"}],
                        "Role": [],
                        "Genre": [],
                    }
                ]
            }
        }

        result = search_plex_library(
            title="The Matrix",
            base_url="http://localhost:32400",
            token="test-token",
            year=1999,
        )

        assert result is not None
        assert result.title == "The Matrix"
        assert result.plex_rating_key == "12345"
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["params"]["query"] == "The Matrix"
        assert call_args[1]["params"]["year"] == 1999

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_search_library_no_results(self, mock_request):
        """Test search with no results returns None."""
        mock_request.return_value = {
            "MediaContainer": {
                "Metadata": []
            }
        }

        result = search_plex_library(
            title="Nonexistent Movie",
            base_url="http://localhost:32400",
            token="test-token",
        )

        assert result is None


class TestGetMovieByRatingKey:
    """Tests for get_movie_by_rating_key orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_movie_success(self, mock_request):
        """Test successful movie metadata retrieval."""
        mock_request.return_value = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "title": "The Matrix",
                        "year": 1999,
                        "ratingKey": "12345",
                        "Guid": [{"id": "tmdb://603"}],
                        "Role": [],
                        "Genre": [],
                    }
                ]
            }
        }

        result = get_movie_by_rating_key(
            rating_key="12345",
            base_url="http://localhost:32400",
            token="test-token",
        )

        assert result.title == "The Matrix"
        assert result.plex_rating_key == "12345"
        
        mock_request.assert_called_once_with(
            "http://localhost:32400/library/metadata/12345",
            "test-token",
        )

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_movie_not_found(self, mock_request):
        """Test that movie not found raises error."""
        mock_request.side_effect = RuntimeError("Plex resource not found")

        with pytest.raises(RuntimeError):
            get_movie_by_rating_key(
                rating_key="99999",
                base_url="http://localhost:32400",
                token="test-token",
            )


class TestExtractFilePath:
    """Tests for extract_file_path function."""

    def test_extract_file_path_success(self):
        """Test extracting file path from movie data."""
        movie_data = {
            "ratingKey": "12345",
            "title": "The Matrix",
            "Media": [
                {
                    "Part": [
                        {"file": "/movies/The Matrix (1999)/The Matrix (1999).mkv"}
                    ]
                }
            ]
        }
        result = extract_file_path(movie_data)
        assert result == "/movies/The Matrix (1999)/The Matrix (1999).mkv"

    def test_extract_file_path_no_media(self):
        """Test that missing Media array returns None."""
        movie_data = {
            "ratingKey": "12345",
            "title": "The Matrix",
        }
        result = extract_file_path(movie_data)
        assert result is None

    def test_extract_file_path_empty_media(self):
        """Test that empty Media array returns None."""
        movie_data = {
            "ratingKey": "12345",
            "title": "The Matrix",
            "Media": []
        }
        result = extract_file_path(movie_data)
        assert result is None

    def test_extract_file_path_no_parts(self):
        """Test that missing Part array returns None."""
        movie_data = {
            "ratingKey": "12345",
            "title": "The Matrix",
            "Media": [{}]
        }
        result = extract_file_path(movie_data)
        assert result is None

    def test_extract_file_path_no_file_field(self):
        """Test that missing file field returns None."""
        movie_data = {
            "ratingKey": "12345",
            "title": "The Matrix",
            "Media": [
                {
                    "Part": [{"id": 123}]  # No 'file' field
                }
            ]
        }
        result = extract_file_path(movie_data)
        assert result is None


class TestExtractFilePathFromResponse:
    """Tests for extract_file_path_from_response function."""

    def test_extract_from_full_response(self):
        """Test extracting file path from full API response."""
        data = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "ratingKey": "12345",
                        "title": "The Matrix",
                        "Media": [
                            {
                                "Part": [
                                    {"file": "/movies/The Matrix (1999)/The Matrix (1999).mkv"}
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        result = extract_file_path_from_response(data)
        assert result == "/movies/The Matrix (1999)/The Matrix (1999).mkv"

    def test_extract_from_empty_response(self):
        """Test extracting from empty response returns None."""
        data = {"MediaContainer": {"Metadata": []}}
        result = extract_file_path_from_response(data)
        assert result is None


class TestGetMovieLibrarySections:
    """Tests for get_movie_library_sections orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_movie_sections_success(self, mock_request):
        """Test successful retrieval of movie library sections."""
        mock_request.return_value = {
            "MediaContainer": {
                "Directory": [
                    {"key": "1", "title": "Movies", "type": "movie"},
                    {"key": "2", "title": "TV Shows", "type": "show"},
                    {"key": "3", "title": "4K Movies", "type": "movie"},
                ]
            }
        }

        result = get_movie_library_sections("http://localhost:32400", "test-token")

        assert len(result) == 2
        assert {"key": "1", "title": "Movies"} in result
        assert {"key": "3", "title": "4K Movies"} in result

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_movie_sections_empty(self, mock_request):
        """Test empty library returns empty list."""
        mock_request.return_value = {
            "MediaContainer": {
                "Directory": []
            }
        }

        result = get_movie_library_sections("http://localhost:32400", "test-token")
        assert result == []


class TestGetAllMoviesWithPaths:
    """Tests for get_all_movies_with_paths orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_movie_library_sections")
    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_all_movies_success(self, mock_request, mock_get_sections):
        """Test successful retrieval of all movies with paths."""
        mock_get_sections.return_value = [{"key": "1", "title": "Movies"}]
        mock_request.return_value = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "ratingKey": "12345",
                        "title": "The Matrix",
                        "Media": [{"Part": [{"file": "/movies/Matrix.mkv"}]}]
                    },
                    {
                        "ratingKey": "12346",
                        "title": "Inception",
                        "Media": [{"Part": [{"file": "/movies/Inception.mkv"}]}]
                    },
                ]
            }
        }

        result = get_all_movies_with_paths("http://localhost:32400", "test-token")

        assert len(result) == 2
        assert ("12345", "The Matrix", "/movies/Matrix.mkv") in result
        assert ("12346", "Inception", "/movies/Inception.mkv") in result

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.make_plex_request")
    def test_get_all_movies_with_section_key(self, mock_request):
        """Test retrieval with specific section key."""
        mock_request.return_value = {
            "MediaContainer": {
                "Metadata": [
                    {
                        "ratingKey": "12345",
                        "title": "The Matrix",
                        "Media": [{"Part": [{"file": "/movies/Matrix.mkv"}]}]
                    }
                ]
            }
        }

        result = get_all_movies_with_paths(
            "http://localhost:32400", "test-token", section_key="1"
        )

        assert len(result) == 1
        mock_request.assert_called_once()
        assert "/library/sections/1/all" in mock_request.call_args[0][0]


class TestFindMovieByFilePath:
    """Tests for find_movie_by_file_path orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_all_movies_with_paths")
    def test_find_movie_by_path_success(self, mock_get_movies):
        """Test finding movie by file path."""
        mock_get_movies.return_value = [
            ("12345", "The Matrix", "/movies/Matrix.mkv"),
            ("12346", "Inception", "/movies/Inception.mkv"),
        ]

        result = find_movie_by_file_path(
            "/movies/Matrix.mkv",
            "http://localhost:32400",
            "test-token",
        )

        assert result == "12345"

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_all_movies_with_paths")
    def test_find_movie_by_path_not_found(self, mock_get_movies):
        """Test that non-existent path returns None."""
        mock_get_movies.return_value = [
            ("12345", "The Matrix", "/movies/Matrix.mkv"),
        ]

        result = find_movie_by_file_path(
            "/movies/NonExistent.mkv",
            "http://localhost:32400",
            "test-token",
        )

        assert result is None

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_all_movies_with_paths")
    def test_find_movie_by_path_with_none_paths(self, mock_get_movies):
        """Test that movies without paths are skipped."""
        mock_get_movies.return_value = [
            ("12345", "No File Movie", None),
            ("12346", "The Matrix", "/movies/Matrix.mkv"),
        ]

        result = find_movie_by_file_path(
            "/movies/Matrix.mkv",
            "http://localhost:32400",
            "test-token",
        )

        assert result == "12346"


class TestBuildFilePathToRatingKeyMap:
    """Tests for build_file_path_to_rating_key_map orchestrator function."""

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_all_movies_with_paths")
    def test_build_map_success(self, mock_get_movies):
        """Test successful map building."""
        mock_get_movies.return_value = [
            ("12345", "The Matrix", "/movies/Matrix.mkv"),
            ("12346", "Inception", "/movies/Inception.mkv"),
            ("12347", "No File", None),  # Should be skipped
        ]

        result = build_file_path_to_rating_key_map(
            "http://localhost:32400",
            "test-token",
        )

        assert len(result) == 2
        # Check that paths are normalized (resolved)
        assert any("12345" in v for v in result.values())
        assert any("12346" in v for v in result.values())

    @patch("src.components.metadata.utilities.plex_client.orchestrate_plex.get_all_movies_with_paths")
    def test_build_map_empty(self, mock_get_movies):
        """Test empty library returns empty map."""
        mock_get_movies.return_value = []

        result = build_file_path_to_rating_key_map(
            "http://localhost:32400",
            "test-token",
        )

        assert result == {}
