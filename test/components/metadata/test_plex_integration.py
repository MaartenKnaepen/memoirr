"""Integration tests for Plex client with real Plex server.

These tests require actual Plex credentials and a running Plex server.
They are skipped by default unless PLEX_INTEGRATION_TEST=1 is set.

To run these tests:
    PLEX_INTEGRATION_TEST=1 uv run pytest test/components/metadata/test_plex_integration.py -v -s

Requirements:
    - PLEX_URL environment variable set (e.g., "http://localhost:32400")
    - PLEX_TOKEN environment variable set (your X-Plex-Token)
    - At least one movie in your Plex library

Adheres to Memoirr testing standards: real-world validation, comprehensive logging.
"""

import os
import pytest

from src.components.metadata.plex_client import PlexClient
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.getenv("PLEX_INTEGRATION_TEST") != "1",
    reason="Integration tests disabled. Set PLEX_INTEGRATION_TEST=1 to run."
)


@pytest.fixture
def plex_client():
    """Create a PlexClient connected to real Plex server."""
    logger.info("Initializing Plex client for integration testing")
    client = PlexClient()
    logger.info(
        "Plex client initialized",
        base_url=client.base_url,
        server_id=client.server_id,
    )
    return client


class TestPlexIntegration:
    """Integration tests with real Plex server."""

    def test_server_connection(self, plex_client):
        """Test that we can connect to Plex server and retrieve server ID."""
        logger.info("Testing Plex server connection")
        
        server_id = plex_client.get_server_id()
        
        assert server_id is not None, "Failed to retrieve server ID"
        assert len(server_id) > 0, "Server ID is empty"
        assert isinstance(server_id, str), "Server ID should be a string"
        
        logger.info("✅ Successfully connected to Plex server", server_id=server_id)
        print(f"\n✅ Server ID: {server_id}")

    def test_search_known_movie(self, plex_client):
        """Test searching for a movie that should exist in most libraries.
        
        This test searches for common movie titles. If your library doesn't have
        any of these, the test will be skipped with a helpful message.
        """
        logger.info("Testing movie search in Plex library")
        
        # Try searching for several common movies
        search_terms = [
            ("The Matrix", 1999),
            ("The Lord of the Rings", 2001),
            ("Star Wars", 1977),
            ("The Godfather", 1972),
            ("Inception", 2010),
        ]
        
        found_movie = None
        for title, year in search_terms:
            logger.info(f"Searching for: {title} ({year})")
            result = plex_client.search(title, year=year)
            
            if result:
                found_movie = result
                logger.info(
                    f"✅ Found movie in library",
                    title=result.title,
                    year=result.year,
                    rating_key=result.plex_rating_key,
                )
                print(f"\n✅ Found: {result.title} ({result.year})")
                print(f"   Rating Key: {result.plex_rating_key}")
                print(f"   TMDB ID: {result.tmdb_id}")
                print(f"   Genres: {', '.join(result.genres) if result.genres else 'N/A'}")
                print(f"   Cast: {len(result.cast)} members")
                break
        
        if not found_movie:
            pytest.skip(
                f"None of the test movies found in your Plex library. "
                f"Searched for: {', '.join([t for t, _ in search_terms])}. "
                f"This is not a failure - just means your library doesn't have these movies."
            )
        
        # Validate the found movie
        assert found_movie.title is not None
        assert found_movie.year > 0
        assert found_movie.plex_rating_key is not None
        assert len(found_movie.plex_rating_key) > 0

    def test_get_metadata_by_rating_key(self, plex_client):
        """Test retrieving metadata for a specific movie by rating key.
        
        This test first searches for a movie, then retrieves its full metadata.
        """
        logger.info("Testing metadata retrieval by rating key")
        
        # First, find a movie
        search_terms = [
            ("The Matrix", 1999),
            ("The Lord of the Rings", 2001),
            ("Star Wars", 1977),
            ("Inception", 2010),
        ]
        
        search_result = None
        for title, year in search_terms:
            search_result = plex_client.search(title, year=year)
            if search_result:
                break
        
        if not search_result:
            pytest.skip("No test movies found in library for metadata test")
        
        # Now get full metadata
        rating_key = search_result.plex_rating_key
        logger.info(f"Fetching metadata for rating key: {rating_key}")
        
        metadata = plex_client.get_metadata(rating_key)
        
        assert metadata is not None
        assert metadata.title == search_result.title
        assert metadata.plex_rating_key == rating_key
        
        logger.info(
            "✅ Successfully retrieved metadata",
            title=metadata.title,
            overview_length=len(metadata.overview) if metadata.overview else 0,
        )
        print(f"\n✅ Retrieved metadata for: {metadata.title}")
        if metadata.overview:
            print(f"   Overview: {metadata.overview[:100]}...")
        print(f"   Genres: {', '.join(metadata.genres) if metadata.genres else 'N/A'}")
        if metadata.cast:
            print(f"   Top cast: {', '.join([c.name for c in metadata.cast[:3]])}")

    def test_generate_deep_link_web(self, plex_client):
        """Test generating a web client deep link."""
        logger.info("Testing web client deep link generation")
        
        # Find a movie first
        search_terms = [
            ("The Matrix", 1999),
            ("The Lord of the Rings", 2001),
            ("Star Wars", 1977),
        ]
        
        movie = None
        for title, year in search_terms:
            movie = plex_client.search(title, year=year)
            if movie:
                break
        
        if not movie:
            pytest.skip("No test movies found in library for deep link test")
        
        # Generate web deep link for 5 minutes into the movie
        timestamp_ms = 5 * 60 * 1000  # 5 minutes
        deep_link = plex_client.get_deep_link(
            rating_key=movie.plex_rating_key,
            start_ms=timestamp_ms,
            client_type="web",
        )
        
        assert deep_link is not None
        assert "https://app.plex.tv/desktop/" in deep_link
        assert plex_client.server_id in deep_link
        assert movie.plex_rating_key in deep_link
        assert "t=300000" in deep_link
        
        logger.info("✅ Generated web deep link", link=deep_link)
        print(f"\n✅ Web Deep Link (5 minutes in):")
        print(f"   {deep_link}")
        print(f"\n   🔗 Click this link to test playback in Plex Web!")

    def test_generate_deep_link_desktop(self, plex_client):
        """Test generating a desktop client deep link."""
        logger.info("Testing desktop client deep link generation")
        
        # Find a movie first
        search_terms = [
            ("The Matrix", 1999),
            ("The Lord of the Rings", 2001),
            ("Star Wars", 1977),
        ]
        
        movie = None
        for title, year in search_terms:
            movie = plex_client.search(title, year=year)
            if movie:
                break
        
        if not movie:
            pytest.skip("No test movies found in library for deep link test")
        
        # Generate desktop deep link for 2 minutes into the movie
        timestamp_ms = 2 * 60 * 1000  # 2 minutes
        deep_link = plex_client.get_deep_link(
            rating_key=movie.plex_rating_key,
            start_ms=timestamp_ms,
            client_type="desktop",
        )
        
        assert deep_link is not None
        assert deep_link.startswith("plex://")
        assert plex_client.server_id in deep_link
        assert movie.plex_rating_key in deep_link
        assert "t=120000" in deep_link
        
        logger.info("✅ Generated desktop deep link", link=deep_link)
        print(f"\n✅ Desktop Deep Link (2 minutes in):")
        print(f"   {deep_link}")
        print(f"\n   🔗 Use this link in Plex desktop app!")

    def test_search_lord_of_rings_movies(self, plex_client):
        """Test searching for Lord of the Rings movies (since they're in your test data)."""
        logger.info("Testing search for Lord of the Rings trilogy")
        
        lotr_movies = [
            ("The Lord of the Rings: The Fellowship of the Ring", 2001),
            ("The Lord of the Rings: The Two Towers", 2002),
            ("The Lord of the Rings: The Return of the King", 2003),
        ]
        
        found_count = 0
        print("\n🔍 Searching for LOTR trilogy in your library:")
        
        for title, year in lotr_movies:
            result = plex_client.search(title, year=year)
            
            if result:
                found_count += 1
                logger.info(
                    f"✅ Found LOTR movie",
                    title=result.title,
                    rating_key=result.plex_rating_key,
                )
                print(f"   ✅ {result.title} ({result.year}) - Rating Key: {result.plex_rating_key}")
            else:
                logger.warning(f"❌ Not found: {title} ({year})")
                print(f"   ❌ {title} ({year}) - Not in library")
        
        if found_count == 0:
            pytest.skip(
                "No LOTR movies found in library. This test was looking for movies "
                "that match the subtitle files in your data/ directory."
            )
        
        print(f"\n   📊 Found {found_count}/3 LOTR movies in your Plex library")
        logger.info(f"Found {found_count} LOTR movies in library", count=found_count)


class TestPlexErrorHandling:
    """Test error handling with real server."""

    def test_search_nonexistent_movie(self, plex_client):
        """Test searching for a movie that definitely doesn't exist."""
        logger.info("Testing search for nonexistent movie")
        
        result = plex_client.search(
            "This Movie Definitely Does Not Exist XYZ123",
            year=9999,
        )
        
        assert result is None
        logger.info("✅ Correctly returned None for nonexistent movie")
        print("\n✅ Correctly handled search for nonexistent movie")

    def test_invalid_rating_key(self, plex_client):
        """Test that invalid rating key raises appropriate error."""
        logger.info("Testing invalid rating key")
        
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            plex_client.get_metadata("999999999")
        
        logger.info("✅ Correctly raised error for invalid rating key", error=str(exc_info.value))
        print(f"\n✅ Correctly raised error for invalid rating key: {exc_info.typename}")
