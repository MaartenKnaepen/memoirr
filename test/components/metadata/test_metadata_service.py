"""Tests for MetadataService component.

Tests the Haystack component that coordinates Plex and TMDB clients
to resolve file paths to complete movie metadata using a "Plex-Only" strategy.

Adheres to Memoirr testing standards: mock at import location, comprehensive coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.components.metadata.metadata_service import MetadataService
from src.components.metadata.utilities.types import MovieMetadata, CastMember


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_movie_metadata():
    """Create a complete MovieMetadata object for testing."""
    return MovieMetadata(
        title="The Matrix",
        year=1999,
        tmdb_id=603,
        radarr_id=None,
        plex_rating_key="12345",
        cast=[
            CastMember(
                name="Keanu Reeves",
                character="Neo",
                tmdb_id=6384,
                profile_path="/path/to/keanu.jpg",
            ),
            CastMember(
                name="Laurence Fishburne",
                character="Morpheus",
                tmdb_id=2975,
                profile_path="/path/to/laurence.jpg",
            ),
        ],
        genres=["Action", "Science Fiction"],
        overview="A computer hacker learns about the true nature of reality.",
    )


@pytest.fixture
def mock_plex_metadata():
    """Create MovieMetadata as returned by Plex (minimal cast)."""
    return MovieMetadata(
        title="The Matrix",
        year=1999,
        tmdb_id=603,
        radarr_id=None,
        plex_rating_key="12345",
        cast=[],
        genres=["Action"],
        overview=None,
    )


@pytest.fixture
def mock_tmdb_metadata():
    """Create MovieMetadata as returned by TMDB (full cast, no Plex ID)."""
    return MovieMetadata(
        title="The Matrix",
        year=1999,
        tmdb_id=603,
        radarr_id=None,
        plex_rating_key=None,
        cast=[
            CastMember(
                name="Keanu Reeves",
                character="Neo",
                tmdb_id=6384,
                profile_path="/path/to/keanu.jpg",
            ),
            CastMember(
                name="Laurence Fishburne",
                character="Morpheus",
                tmdb_id=2975,
                profile_path="/path/to/laurence.jpg",
            ),
        ],
        genres=["Action", "Science Fiction"],
        overview="A computer hacker learns about the true nature of reality.",
    )


# ============================================================================
# Tests for MetadataService Initialization
# ============================================================================


class TestMetadataServiceInit:
    """Tests for MetadataService initialization."""

    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_init_with_defaults(
        self, mock_get_settings, mock_tmdb, mock_plex
    ):
        """Test initialization with default settings."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        service = MetadataService()

        assert service.top_cast == 20
        mock_tmdb.assert_called_once()
        mock_plex.assert_called_once_with(base_url=None, token=None)

    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_init_with_custom_params(
        self, mock_get_settings, mock_tmdb, mock_plex
    ):
        """Test initialization with custom parameters."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        service = MetadataService(
            top_cast=10,
            plex_url="http://custom-plex:32400",
            plex_token="custom-plex-token",
        )

        assert service.top_cast == 10
        mock_plex.assert_called_once_with(
            base_url="http://custom-plex:32400",
            token="custom-plex-token",
        )

    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_init_tmdb_missing_key_raises(
        self, mock_get_settings, mock_tmdb, mock_plex
    ):
        """Test that missing TMDB API key raises ValueError."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_tmdb.side_effect = ValueError("TMDB_API_KEY is required")

        with pytest.raises(ValueError, match="TMDB_API_KEY"):
            MetadataService()


# ============================================================================
# Tests for MetadataService.run() - Happy Paths
# ============================================================================


class TestMetadataServiceRun:
    """Tests for MetadataService.run() method."""

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_plex_hit_tmdb_hit(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
        mock_movie_metadata,
    ):
        """Test happy path: File found in Plex, TMDB returns full metadata."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.return_value = mock_movie_metadata

        service = MetadataService()
        result = service.run("/movies/The Matrix (1999)/The.Matrix.1999.mkv")

        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata.title == "The Matrix"
        assert metadata.year == 1999
        assert metadata.tmdb_id == 603
        assert metadata.plex_rating_key == "12345"
        assert len(metadata.cast) == 2
        assert metadata.cast[0].name == "Keanu Reeves"

        mock_orchestrate.assert_called_once()
        call_kwargs = mock_orchestrate.call_args[1]
        assert call_kwargs["top_cast"] == 20

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_custom_top_cast(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
        mock_movie_metadata,
    ):
        """Test that custom top_cast parameter is passed through."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.return_value = mock_movie_metadata

        service = MetadataService(top_cast=5)
        service.run("/movies/test.mkv")

        mock_orchestrate.assert_called_once()
        call_kwargs = mock_orchestrate.call_args[1]
        assert call_kwargs["top_cast"] == 5


# ============================================================================
# Tests for MetadataService.run() - Error Cases (Plex-Only Fail Fast)
# ============================================================================


class TestMetadataServiceRunErrors:
    """Tests for MetadataService.run() error handling with Plex-Only strategy."""

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_plex_miss_raises_value_error(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
    ):
        """Test that ValueError is raised when file is not found in Plex (fail fast)."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.side_effect = ValueError(
            "File not found in Plex library: /movies/unknown_file.mkv"
        )

        service = MetadataService()

        with pytest.raises(ValueError, match="File not found in Plex library"):
            service.run("/movies/unknown_file.mkv")

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_connection_error_propagated(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
    ):
        """Test that ConnectionError is propagated when Plex is unreachable."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.side_effect = ConnectionError("Plex server unreachable")

        service = MetadataService()

        with pytest.raises(ConnectionError, match="Plex server unreachable"):
            service.run("/movies/test.mkv")

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_runtime_error_propagated(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
    ):
        """Test that RuntimeError is propagated when API fails."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.side_effect = RuntimeError("TMDB API returned 500")

        service = MetadataService()

        with pytest.raises(RuntimeError, match="TMDB API returned 500"):
            service.run("/movies/test.mkv")


# ============================================================================
# Tests for Haystack Component Integration
# ============================================================================


class TestMetadataServiceHaystackIntegration:
    """Tests for Haystack component integration."""

    @patch("src.components.metadata.metadata_service.orchestrate_metadata")
    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_output_types_match_return_keys(
        self,
        mock_get_settings,
        mock_tmdb_class,
        mock_plex_class,
        mock_orchestrate,
        mock_movie_metadata,
    ):
        """Test that return dict keys match @component.output_types."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_orchestrate.return_value = mock_movie_metadata

        service = MetadataService()
        result = service.run("/movies/test.mkv")

        # Verify the return dict has exactly the keys declared in output_types
        assert set(result.keys()) == {"metadata"}
        assert isinstance(result["metadata"], MovieMetadata)

    @patch("src.components.metadata.metadata_service.PlexClient")
    @patch("src.components.metadata.metadata_service.TmdbClient")
    @patch("src.components.metadata.metadata_service.get_settings")
    def test_component_decorator_applied(
        self, mock_get_settings, mock_tmdb, mock_plex
    ):
        """Test that the component decorator is properly applied."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        # Create instance and check that it has Haystack component attributes
        service = MetadataService()
        assert hasattr(service, "__haystack_output__")
        output_sockets = getattr(service, "__haystack_output__")
        assert "metadata" in output_sockets._sockets_dict
