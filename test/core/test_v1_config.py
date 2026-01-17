"""Tests for V1 multimodal configuration settings.

Tests the new configuration fields added for Phase 1 of the V1 multimodal
implementation, including metadata API settings, vision pipeline settings,
and speaker tagging configuration.
"""

import os
from unittest.mock import patch

import pytest

from src.core.config import get_settings


class TestV1MetadataConfig:
    """Test metadata API configuration settings."""

    def test_tmdb_defaults(self):
        """Test TMDB configuration with default values."""
        settings = get_settings()
        
        assert settings.tmdb_api_key is None
        assert settings.tmdb_base_url == "https://api.themoviedb.org/3"

    def test_tmdb_env_override(self):
        """Test TMDB configuration can be overridden via environment variables."""
        with patch.dict(os.environ, {
            "TMDB_API_KEY": "test_tmdb_key_123",
            "TMDB_BASE_URL": "https://custom.tmdb.url"
        }):
            # Clear cache to force reload
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.tmdb_api_key == "test_tmdb_key_123"
            assert settings.tmdb_base_url == "https://custom.tmdb.url"
            
            # Cleanup
            get_settings.cache_clear()

    def test_radarr_defaults(self):
        """Test Radarr configuration with default values."""
        settings = get_settings()
        
        assert settings.radarr_url is None
        assert settings.radarr_api_key is None

    def test_radarr_env_override(self):
        """Test Radarr configuration can be overridden via environment variables."""
        with patch.dict(os.environ, {
            "RADARR_URL": "http://localhost:7878",
            "RADARR_API_KEY": "test_radarr_key_456"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.radarr_url == "http://localhost:7878"
            assert settings.radarr_api_key == "test_radarr_key_456"
            
            get_settings.cache_clear()

    def test_plex_defaults(self):
        """Test Plex configuration with default values."""
        settings = get_settings()
        
        assert settings.plex_url is None
        assert settings.plex_token is None

    def test_plex_env_override(self):
        """Test Plex configuration can be overridden via environment variables."""
        with patch.dict(os.environ, {
            "PLEX_URL": "http://localhost:32400",
            "PLEX_TOKEN": "test_plex_token_789"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.plex_url == "http://localhost:32400"
            assert settings.plex_token == "test_plex_token_789"
            
            get_settings.cache_clear()


class TestV1VisionConfig:
    """Test vision pipeline configuration settings."""

    def test_scene_detection_defaults(self):
        """Test scene detection configuration with default values."""
        settings = get_settings()
        
        assert settings.scene_detect_threshold == 27.0
        assert settings.scene_min_duration_sec == 1.5
        assert settings.scene_merge_threshold_sec == 5.0
        assert settings.keyframes_per_scene == 3

    def test_scene_detection_env_override(self):
        """Test scene detection configuration can be overridden."""
        with patch.dict(os.environ, {
            "SCENE_DETECT_THRESHOLD": "30.0",
            "SCENE_MIN_DURATION_SEC": "2.0",
            "SCENE_MERGE_THRESHOLD_SEC": "10.0",
            "KEYFRAMES_PER_SCENE": "5"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.scene_detect_threshold == 30.0
            assert settings.scene_min_duration_sec == 2.0
            assert settings.scene_merge_threshold_sec == 10.0
            assert settings.keyframes_per_scene == 5
            
            get_settings.cache_clear()

    def test_face_recognition_defaults(self):
        """Test face recognition configuration with default values."""
        settings = get_settings()
        
        assert settings.face_similarity_threshold == 0.6
        assert settings.face_min_confidence == 0.5
        assert settings.insightface_model == "buffalo_l"

    def test_face_recognition_env_override(self):
        """Test face recognition configuration can be overridden."""
        with patch.dict(os.environ, {
            "FACE_SIMILARITY_THRESHOLD": "0.7",
            "FACE_MIN_CONFIDENCE": "0.6",
            "INSIGHTFACE_MODEL": "buffalo_s"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.face_similarity_threshold == 0.7
            assert settings.face_min_confidence == 0.6
            assert settings.insightface_model == "buffalo_s"
            
            get_settings.cache_clear()

    def test_vlm_defaults(self):
        """Test Vision-Language Model configuration with default values."""
        settings = get_settings()
        
        assert settings.vlm_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
        assert settings.vlm_quantization_bits == 4
        assert settings.vlm_max_new_tokens == 512

    def test_vlm_env_override(self):
        """Test VLM configuration can be overridden."""
        with patch.dict(os.environ, {
            "VLM_MODEL_ID": "microsoft/Florence-2-large",
            "VLM_QUANTIZATION_BITS": "8",
            "VLM_MAX_NEW_TOKENS": "1024"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.vlm_model_id == "microsoft/Florence-2-large"
            assert settings.vlm_quantization_bits == 8
            assert settings.vlm_max_new_tokens == 1024
            
            get_settings.cache_clear()


class TestV1SpeakerConfig:
    """Test speaker tagging configuration settings."""

    def test_speaker_tagging_defaults(self):
        """Test speaker tagging configuration with default values."""
        settings = get_settings()
        
        assert settings.speaker_confidence_threshold == 0.8
        assert settings.speaker_sliding_window == 20
        assert settings.speaker_model == "llama-3.3-70b-versatile"

    def test_speaker_tagging_env_override(self):
        """Test speaker tagging configuration can be overridden."""
        with patch.dict(os.environ, {
            "SPEAKER_CONFIDENCE_THRESHOLD": "0.9",
            "SPEAKER_SLIDING_WINDOW": "30",
            "SPEAKER_MODEL": "llama-3.1-70b-versatile"
        }):
            get_settings.cache_clear()
            settings = get_settings()
            
            assert settings.speaker_confidence_threshold == 0.9
            assert settings.speaker_sliding_window == 30
            assert settings.speaker_model == "llama-3.1-70b-versatile"
            
            get_settings.cache_clear()


class TestV1ConfigIntegration:
    """Test integration and edge cases for V1 configuration."""

    def test_all_v1_settings_accessible(self):
        """Test that all V1 settings can be accessed without errors."""
        settings = get_settings()
        
        # Metadata settings
        _ = settings.tmdb_api_key
        _ = settings.tmdb_base_url
        _ = settings.radarr_url
        _ = settings.radarr_api_key
        _ = settings.plex_url
        _ = settings.plex_token
        
        # Vision settings
        _ = settings.scene_detect_threshold
        _ = settings.scene_min_duration_sec
        _ = settings.scene_merge_threshold_sec
        _ = settings.keyframes_per_scene
        _ = settings.face_similarity_threshold
        _ = settings.face_min_confidence
        _ = settings.insightface_model
        _ = settings.vlm_model_id
        _ = settings.vlm_quantization_bits
        _ = settings.vlm_max_new_tokens
        
        # Speaker settings
        _ = settings.speaker_confidence_threshold
        _ = settings.speaker_sliding_window
        _ = settings.speaker_model

    def test_optional_fields_are_none_by_default(self):
        """Test that optional API keys are None when not configured."""
        settings = get_settings()
        
        # These should be None when not configured (not raise errors)
        assert settings.tmdb_api_key is None
        assert settings.radarr_url is None
        assert settings.radarr_api_key is None
        assert settings.plex_url is None
        assert settings.plex_token is None

    def test_numeric_types_correct(self):
        """Test that numeric settings have correct types."""
        settings = get_settings()
        
        # Float types
        assert isinstance(settings.scene_detect_threshold, float)
        assert isinstance(settings.scene_min_duration_sec, float)
        assert isinstance(settings.scene_merge_threshold_sec, float)
        assert isinstance(settings.face_similarity_threshold, float)
        assert isinstance(settings.face_min_confidence, float)
        assert isinstance(settings.speaker_confidence_threshold, float)
        
        # Int types
        assert isinstance(settings.keyframes_per_scene, int)
        assert isinstance(settings.vlm_quantization_bits, int)
        assert isinstance(settings.vlm_max_new_tokens, int)
        assert isinstance(settings.speaker_sliding_window, int)

    def test_string_types_correct(self):
        """Test that string settings have correct types."""
        settings = get_settings()
        
        assert isinstance(settings.tmdb_base_url, str)
        assert isinstance(settings.insightface_model, str)
        assert isinstance(settings.vlm_model_id, str)
        assert isinstance(settings.speaker_model, str)
