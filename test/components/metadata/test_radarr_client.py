"""Tests for RadarrClient component.

Tests the component-level integration including initialization and configuration.
Adheres to Memoirr testing standards: mock at import location, proper fixtures.
"""

import pytest
from unittest.mock import patch, Mock

from src.components.metadata.radarr_client import RadarrClient
from src.components.metadata.utilities.types import MovieMetadata


# ============================================================================
# Tests: RadarrClient Component
# ============================================================================

class TestRadarrClientInitialization:
    """Tests for RadarrClient initialization and configuration."""

    @patch("src.components.metadata.radarr_client.get_settings")
    def test_initialization_with_env_config(self, mock_settings):
        """Test initialization using environment configuration."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="env-api-key",
        )

        client = RadarrClient()

        assert client.radarr_url == "http://localhost:7878"
        assert client.radarr_api_key == "env-api-key"

    @patch("src.components.metadata.radarr_client.get_settings")
    def test_initialization_with_override(self, mock_settings):
        """Test initialization with parameter overrides."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="env-api-key",
        )

        client = RadarrClient(
            radarr_url="http://custom:8080",
            radarr_api_key="custom-key",
        )

        assert client.radarr_url == "http://custom:8080"
        assert client.radarr_api_key == "custom-key"

    @patch("src.components.metadata.radarr_client.get_settings")
    def test_initialization_missing_url(self, mock_settings):
        """Test error when Radarr URL is not configured."""
        mock_settings.return_value = Mock(
            radarr_url=None,
            radarr_api_key="test-key",
        )

        with pytest.raises(ValueError, match="Radarr URL is required"):
            RadarrClient()

    @patch("src.components.metadata.radarr_client.get_settings")
    def test_initialization_missing_api_key(self, mock_settings):
        """Test error when Radarr API key is not configured."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key=None,
        )

        with pytest.raises(ValueError, match="Radarr API key is required"):
            RadarrClient()

    @patch("src.components.metadata.radarr_client.get_settings")
    def test_url_trailing_slash_removal(self, mock_settings):
        """Test that trailing slashes are removed from URL."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878/",
            radarr_api_key="test-key",
        )

        client = RadarrClient()

        assert client.radarr_url == "http://localhost:7878"


class TestRadarrClientMethods:
    """Tests for RadarrClient public methods."""

    @patch("src.components.metadata.radarr_client.get_settings")
    @patch("src.components.metadata.radarr_client.get_all_movies")
    def test_get_all_movies(self, mock_get_all, mock_settings):
        """Test get_all_movies delegates to orchestrator."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="test-key",
        )

        mock_movie = MovieMetadata(
            title="Test Movie",
            year=2000,
            tmdb_id=123,
            radarr_id=1,
            plex_rating_key=None,
            cast=[],
        )
        mock_get_all.return_value = [mock_movie]

        client = RadarrClient()
        movies = client.get_all_movies()

        assert len(movies) == 1
        assert movies[0].title == "Test Movie"
        mock_get_all.assert_called_once_with("http://localhost:7878", "test-key")

    @patch("src.components.metadata.radarr_client.get_settings")
    @patch("src.components.metadata.radarr_client.orchestrate_get_movie_by_tmdb_id")
    def test_get_movie_by_tmdb_id(self, mock_get_by_tmdb, mock_settings):
        """Test get_movie_by_tmdb_id delegates to orchestrator."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="test-key",
        )

        mock_movie = MovieMetadata(
            title="Test Movie",
            year=2000,
            tmdb_id=123,
            radarr_id=1,
            plex_rating_key=None,
            cast=[],
        )
        mock_get_by_tmdb.return_value = mock_movie

        client = RadarrClient()
        movie = client.get_movie_by_tmdb_id(tmdb_id=123)

        assert movie.title == "Test Movie"
        assert movie.tmdb_id == 123
        mock_get_by_tmdb.assert_called_once_with(
            tmdb_id=123,
            base_url="http://localhost:7878",
            api_key="test-key",
        )

    @patch("src.components.metadata.radarr_client.get_settings")
    @patch("src.components.metadata.radarr_client.get_all_movies")
    def test_get_movie_by_radarr_id(self, mock_get_all, mock_settings):
        """Test get_movie_by_radarr_id filters by Radarr ID."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="test-key",
        )

        mock_movies = [
            MovieMetadata(
                title="Movie 1",
                year=2000,
                tmdb_id=123,
                radarr_id=1,
                plex_rating_key=None,
                cast=[],
            ),
            MovieMetadata(
                title="Movie 2",
                year=2001,
                tmdb_id=124,
                radarr_id=2,
                plex_rating_key=None,
                cast=[],
            ),
        ]
        mock_get_all.return_value = mock_movies

        client = RadarrClient()
        movie = client.get_movie_by_radarr_id(radarr_id=2)

        assert movie is not None
        assert movie.title == "Movie 2"
        assert movie.radarr_id == 2

    @patch("src.components.metadata.radarr_client.get_settings")
    @patch("src.components.metadata.radarr_client.get_all_movies")
    def test_get_movie_by_radarr_id_not_found(self, mock_get_all, mock_settings):
        """Test get_movie_by_radarr_id returns None when not found."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="test-key",
        )

        mock_get_all.return_value = []

        client = RadarrClient()
        movie = client.get_movie_by_radarr_id(radarr_id=999)

        assert movie is None

    @patch("src.components.metadata.radarr_client.get_settings")
    @patch("src.components.metadata.radarr_client.get_all_movies")
    def test_connection_error_propagates(self, mock_get_all, mock_settings):
        """Test that connection errors from orchestrator are propagated."""
        mock_settings.return_value = Mock(
            radarr_url="http://localhost:7878",
            radarr_api_key="test-key",
        )

        mock_get_all.side_effect = ConnectionError("Cannot connect to Radarr")

        client = RadarrClient()
        with pytest.raises(ConnectionError, match="Cannot connect to Radarr"):
            client.get_all_movies()
