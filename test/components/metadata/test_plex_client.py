"""Tests for PlexClient component.

Tests the high-level PlexClient wrapper class.
Adheres to Memoirr testing standards: mock at import location, comprehensive coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.components.metadata.plex_client import PlexClient
from src.components.metadata.utilities.types import MovieMetadata


class TestPlexClientInit:
    """Tests for PlexClient initialization."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_init_with_settings(self, mock_get_identity, mock_get_settings):
        """Test initialization using settings."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"

        client = PlexClient()

        assert client.base_url == "http://localhost:32400"
        assert client.token == "test-token"
        assert client.server_id == "server-id-123"
        mock_get_identity.assert_called_once_with("http://localhost:32400", "test-token")

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_init_with_explicit_params(self, mock_get_identity, mock_get_settings):
        """Test initialization with explicit parameters."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://default:32400"
        mock_settings.plex_token = "default-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"

        client = PlexClient(base_url="http://custom:32400", token="custom-token")

        assert client.base_url == "http://custom:32400"
        assert client.token == "custom-token"
        assert client.server_id == "server-id-123"
        mock_get_identity.assert_called_once_with("http://custom:32400", "custom-token")

    @patch("src.components.metadata.plex_client.get_settings")
    def test_init_missing_url_raises_error(self, mock_get_settings):
        """Test that missing URL raises ValueError."""
        mock_settings = Mock()
        mock_settings.plex_url = None
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="Plex URL must be provided"):
            PlexClient()

    @patch("src.components.metadata.plex_client.get_settings")
    def test_init_missing_token_raises_error(self, mock_get_settings):
        """Test that missing token raises ValueError."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="Plex token must be provided"):
            PlexClient()

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_init_server_id_fetch_fails_gracefully(self, mock_get_identity, mock_get_settings):
        """Test that initialization continues even if server ID fetch fails."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.side_effect = ConnectionError("Cannot connect")

        client = PlexClient()

        assert client.base_url == "http://localhost:32400"
        assert client.token == "test-token"
        assert client.server_id is None  # Failed to fetch but didn't crash


class TestPlexClientSearch:
    """Tests for PlexClient.search method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.search_plex_library")
    def test_search_success(self, mock_search, mock_get_identity, mock_get_settings):
        """Test successful movie search."""
        # Setup
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"

        mock_metadata = MovieMetadata(
            title="The Matrix",
            year=1999,
            tmdb_id=603,
            radarr_id=None,
            plex_rating_key="12345",
            cast=[],
            genres=["Action", "Sci-Fi"],
            overview="A great movie",
        )
        mock_search.return_value = mock_metadata

        # Execute
        client = PlexClient()
        result = client.search("The Matrix", year=1999)

        # Assert
        assert result == mock_metadata
        mock_search.assert_called_once_with(
            title="The Matrix",
            base_url="http://localhost:32400",
            token="test-token",
            year=1999,
        )

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.search_plex_library")
    def test_search_no_results(self, mock_search, mock_get_identity, mock_get_settings):
        """Test search with no results."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_search.return_value = None

        client = PlexClient()
        result = client.search("Nonexistent Movie")

        assert result is None

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.search_plex_library")
    def test_search_connection_error(self, mock_search, mock_get_identity, mock_get_settings):
        """Test that connection errors are propagated."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_search.side_effect = ConnectionError("Cannot connect")

        client = PlexClient()
        
        with pytest.raises(ConnectionError):
            client.search("The Matrix")


class TestPlexClientGetMetadata:
    """Tests for PlexClient.get_metadata method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.get_movie_by_rating_key")
    def test_get_metadata_success(self, mock_get_movie, mock_get_identity, mock_get_settings):
        """Test successful metadata retrieval."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"

        mock_metadata = MovieMetadata(
            title="The Matrix",
            year=1999,
            tmdb_id=603,
            radarr_id=None,
            plex_rating_key="12345",
            cast=[],
            genres=[],
        )
        mock_get_movie.return_value = mock_metadata

        client = PlexClient()
        result = client.get_metadata("12345")

        assert result == mock_metadata
        mock_get_movie.assert_called_once_with(
            rating_key="12345",
            base_url="http://localhost:32400",
            token="test-token",
        )


class TestPlexClientGetDeepLink:
    """Tests for PlexClient.get_deep_link method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.generate_deep_link")
    def test_get_deep_link_with_cached_server_id(self, mock_generate, mock_get_identity, mock_get_settings):
        """Test deep link generation with cached server ID."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_generate.return_value = "https://app.plex.tv/desktop/#!/server/server-id-123/details?key=%2Flibrary%2Fmetadata%2F12345&t=300000"

        client = PlexClient()
        result = client.get_deep_link("12345", 300000, client_type="web")

        assert "app.plex.tv" in result
        mock_generate.assert_called_once_with(
            server_id="server-id-123",
            rating_key="12345",
            timestamp_ms=300000,
            client_type="web",
        )
        # get_server_identity should only be called once during init
        assert mock_get_identity.call_count == 1

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.generate_deep_link")
    def test_get_deep_link_fetches_server_id_if_not_cached(self, mock_generate, mock_get_identity, mock_get_settings):
        """Test that server ID is fetched on demand if not cached."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        
        # First call fails (during init), second succeeds (during get_deep_link)
        mock_get_identity.side_effect = [
            ConnectionError("Cannot connect"),  # Init fails
            "server-id-123",  # get_deep_link succeeds
        ]
        
        mock_generate.return_value = "plex://server/server-id-123/library/metadata/12345?t=300000"

        client = PlexClient()
        assert client.server_id is None  # Failed during init
        
        result = client.get_deep_link("12345", 300000, client_type="desktop")

        assert "plex://" in result
        # get_server_identity should be called twice: once during init, once during get_deep_link
        assert mock_get_identity.call_count == 2
        mock_generate.assert_called_once_with(
            server_id="server-id-123",
            rating_key="12345",
            timestamp_ms=300000,
            client_type="desktop",
        )

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_get_deep_link_server_id_fetch_fails(self, mock_get_identity, mock_get_settings):
        """Test that connection error is raised if server ID cannot be fetched."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.side_effect = ConnectionError("Cannot connect")

        client = PlexClient()
        
        with pytest.raises(ConnectionError):
            client.get_deep_link("12345", 300000)


class TestPlexClientGetServerId:
    """Tests for PlexClient.get_server_id method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_get_server_id_returns_cached_value(self, mock_get_identity, mock_get_settings):
        """Test that cached server ID is returned."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"

        client = PlexClient()
        result = client.get_server_id()

        assert result == "server-id-123"
        # Should only be called once during init
        assert mock_get_identity.call_count == 1

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    def test_get_server_id_fetches_if_not_cached(self, mock_get_identity, mock_get_settings):
        """Test that server ID is fetched if not cached."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        
        # First call fails, second succeeds
        mock_get_identity.side_effect = [
            ConnectionError("Cannot connect"),
            "server-id-123",
        ]

        client = PlexClient()
        assert client.server_id is None
        
        result = client.get_server_id()

        assert result == "server-id-123"
        assert client.server_id == "server-id-123"
        assert mock_get_identity.call_count == 2


class TestPlexClientFindByFilePath:
    """Tests for PlexClient.find_by_file_path method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.find_movie_by_file_path")
    def test_find_by_file_path_success(self, mock_find, mock_get_identity, mock_get_settings):
        """Test successful file path lookup."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_find.return_value = "12345"

        client = PlexClient()
        result = client.find_by_file_path("/movies/Matrix.mkv")

        assert result == "12345"
        mock_find.assert_called_once_with(
            file_path="/movies/Matrix.mkv",
            base_url="http://localhost:32400",
            token="test-token",
        )

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.find_movie_by_file_path")
    def test_find_by_file_path_not_found(self, mock_find, mock_get_identity, mock_get_settings):
        """Test file path not found returns None."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_find.return_value = None

        client = PlexClient()
        result = client.find_by_file_path("/movies/NonExistent.mkv")

        assert result is None


class TestPlexClientBuildPathMap:
    """Tests for PlexClient.build_path_map method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.build_file_path_to_rating_key_map")
    def test_build_path_map_success(self, mock_build_map, mock_get_identity, mock_get_settings):
        """Test successful path map building."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_build_map.return_value = {
            "/movies/Matrix.mkv": "12345",
            "/movies/Inception.mkv": "12346",
        }

        client = PlexClient()
        result = client.build_path_map()

        assert len(result) == 2
        assert result["/movies/Matrix.mkv"] == "12345"
        mock_build_map.assert_called_once_with(
            base_url="http://localhost:32400",
            token="test-token",
        )


class TestPlexClientGetAllMovies:
    """Tests for PlexClient.get_all_movies method."""

    @patch("src.components.metadata.plex_client.get_settings")
    @patch("src.components.metadata.plex_client.get_server_identity")
    @patch("src.components.metadata.plex_client.get_all_movies_with_paths")
    def test_get_all_movies_success(self, mock_get_movies, mock_get_identity, mock_get_settings):
        """Test successful retrieval of all movies."""
        mock_settings = Mock()
        mock_settings.plex_url = "http://localhost:32400"
        mock_settings.plex_token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_get_identity.return_value = "server-id-123"
        mock_get_movies.return_value = [
            ("12345", "The Matrix", "/movies/Matrix.mkv"),
            ("12346", "Inception", "/movies/Inception.mkv"),
        ]

        client = PlexClient()
        result = client.get_all_movies()

        assert len(result) == 2
        assert ("12345", "The Matrix", "/movies/Matrix.mkv") in result
        mock_get_movies.assert_called_once_with(
            base_url="http://localhost:32400",
            token="test-token",
        )
