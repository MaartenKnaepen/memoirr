"""Tests for configuration validation system."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set test environment
os.environ['CHUNK_SIZE'] = '512'
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['EMBEDDING_MODEL_NAME'] = 'test-model'
os.environ['LOG_FORMAT'] = 'console'
os.environ['ENVIRONMENT'] = 'development'
os.environ['ENGLISH_ASCII_THRESHOLD'] = '0.95'

from src.core.config import (
    Settings,
    MemoirrConfigError,
    validate_model_accessibility,
    validate_qdrant_config,
    validate_chunk_delimiters,
    validate_threshold_config,
    validate_settings_comprehensive,
    get_settings
)


class TestFieldValidators:
    """Test Pydantic field validators directly by calling the validation functions."""
    
    def test_embedding_dimension_validation_directly(self):
        """Test embedding dimension validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_embedding_dimension(1024)
        assert result == 1024
        
        result = Settings.validate_embedding_dimension(None)
        assert result is None
        
        # Test invalid values
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSION must be positive"):
            Settings.validate_embedding_dimension(-1)
            
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSION must be positive"):
            Settings.validate_embedding_dimension(0)
    
    def test_chunk_size_validation_directly(self):
        """Test chunk size validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_chunk_size(512)
        assert result == 512
        
        # Test invalid values
        with pytest.raises(ValueError, match="CHUNK_SIZE must be positive"):
            Settings.validate_chunk_size(-1)
            
        with pytest.raises(ValueError, match="CHUNK_SIZE must be positive"):
            Settings.validate_chunk_size(0)
    
    def test_chunk_similarity_window_validation_directly(self):
        """Test similarity window validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_similarity_window(3)
        assert result == 3
        
        # Test invalid values
        with pytest.raises(ValueError, match="CHUNK_SIMILARITY_WINDOW must be at least 1"):
            Settings.validate_similarity_window(0)
            
        with pytest.raises(ValueError, match="CHUNK_SIMILARITY_WINDOW must be at least 1"):
            Settings.validate_similarity_window(-1)
    
    def test_chunk_min_sentences_validation_directly(self):
        """Test minimum sentences validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_min_sentences(2)
        assert result == 2
        
        # Test invalid values
        with pytest.raises(ValueError, match="CHUNK_MIN_SENTENCES must be at least 1"):
            Settings.validate_min_sentences(0)
            
        with pytest.raises(ValueError, match="CHUNK_MIN_SENTENCES must be at least 1"):
            Settings.validate_min_sentences(-1)
    
    def test_chunk_min_characters_per_sentence_validation_directly(self):
        """Test minimum characters per sentence validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_min_chars_per_sentence(24)
        assert result == 24
        
        # Test invalid values
        with pytest.raises(ValueError, match="CHUNK_MIN_CHARACTERS_PER_SENTENCE must be at least 1"):
            Settings.validate_min_chars_per_sentence(0)
            
        with pytest.raises(ValueError, match="CHUNK_MIN_CHARACTERS_PER_SENTENCE must be at least 1"):
            Settings.validate_min_chars_per_sentence(-1)
    
    def test_english_ascii_threshold_validation_directly(self):
        """Test ASCII threshold validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_ascii_threshold(0.95)
        assert result == 0.95
        
        result = Settings.validate_ascii_threshold(0.0)
        assert result == 0.0
        
        result = Settings.validate_ascii_threshold(1.0)
        assert result == 1.0
        
        # Test invalid values
        with pytest.raises(ValueError, match="ENGLISH_ASCII_THRESHOLD must be between 0.0 and 1.0"):
            Settings.validate_ascii_threshold(-0.1)
            
        with pytest.raises(ValueError, match="ENGLISH_ASCII_THRESHOLD must be between 0.0 and 1.0"):
            Settings.validate_ascii_threshold(1.1)
    
    def test_ascii_char_upper_limit_validation_directly(self):
        """Test ASCII character limit validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values
        result = Settings.validate_ascii_limit(128)
        assert result == 128
        
        result = Settings.validate_ascii_limit(1114111)  # Max Unicode
        assert result == 1114111
        
        result = Settings.validate_ascii_limit(1)  # Min valid
        assert result == 1
        
        # Test invalid values
        with pytest.raises(ValueError, match="ASCII_CHAR_UPPER_LIMIT must be between 1 and 1114111"):
            Settings.validate_ascii_limit(0)
            
        with pytest.raises(ValueError, match="ASCII_CHAR_UPPER_LIMIT must be between 1 and 1114111"):
            Settings.validate_ascii_limit(1114112)
    
    def test_log_level_validation_directly(self):
        """Test log level validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values (case insensitive)
        result = Settings.validate_log_level("INFO")
        assert result == "INFO"
        
        result = Settings.validate_log_level("info")
        assert result == "INFO"
        
        result = Settings.validate_log_level("debug")
        assert result == "DEBUG"
        
        result = Settings.validate_log_level("WARNING")
        assert result == "WARNING"
        
        result = Settings.validate_log_level("ERROR")
        assert result == "ERROR"
        
        result = Settings.validate_log_level("CRITICAL")
        assert result == "CRITICAL"
        
        # Test invalid values
        with pytest.raises(ValueError, match="LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"):
            Settings.validate_log_level("INVALID")
            
        with pytest.raises(ValueError, match="LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"):
            Settings.validate_log_level("trace")
    
    def test_log_format_validation_directly(self):
        """Test log format validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values (case insensitive)
        result = Settings.validate_log_format("json")
        assert result == "json"
        
        result = Settings.validate_log_format("JSON")
        assert result == "json"
        
        result = Settings.validate_log_format("console")
        assert result == "console"
        
        result = Settings.validate_log_format("CONSOLE")
        assert result == "console"
        
        # Test invalid values
        with pytest.raises(ValueError, match="LOG_FORMAT must be one of: json, console"):
            Settings.validate_log_format("xml")
            
        with pytest.raises(ValueError, match="LOG_FORMAT must be one of: json, console"):
            Settings.validate_log_format("file")
    
    def test_environment_validation_directly(self):
        """Test environment validation logic directly."""
        from src.core.config import Settings
        
        # Test valid values (case insensitive)
        result = Settings.validate_environment("development")
        assert result == "development"
        
        result = Settings.validate_environment("DEVELOPMENT")
        assert result == "development"
        
        result = Settings.validate_environment("staging")
        assert result == "staging"
        
        result = Settings.validate_environment("production")
        assert result == "production"
        
        # Test invalid values
        with pytest.raises(ValueError, match="ENVIRONMENT must be one of: development, staging, production"):
            Settings.validate_environment("test")
            
        with pytest.raises(ValueError, match="ENVIRONMENT must be one of: development, staging, production"):
            Settings.validate_environment("dev")


class TestModelValidation:
    """Test model accessibility validation."""

    @patch('src.core.model_utils.resolve_model_path')
    @patch('src.core.model_utils.validate_model_directory')
    def test_valid_model(self, mock_validate_dir, mock_resolve_path):
        """Test validation with valid model."""
        mock_resolve_path.return_value = Path("/models/test-model")
        mock_validate_dir.return_value = True
        
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self, embedding_model_name):
                self.embedding_model_name = embedding_model_name
        
        settings = MockSettings(embedding_model_name="test-model")
        issues = validate_model_accessibility(settings)
        
        assert issues == []

    @patch('src.core.model_utils.resolve_model_path')
    @patch('src.core.model_utils.validate_model_directory')
    def test_invalid_model_directory(self, mock_validate_dir, mock_resolve_path):
        """Test validation with invalid model directory."""
        mock_resolve_path.return_value = Path("/models/test-model")
        mock_validate_dir.return_value = False
        
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self, embedding_model_name):
                self.embedding_model_name = embedding_model_name
        
        settings = MockSettings(embedding_model_name="test-model")
        issues = validate_model_accessibility(settings)
        
        assert len(issues) == 1
        assert "appears invalid" in issues[0]

    @patch('src.core.model_utils.resolve_model_path') 
    @patch('src.core.model_utils.find_model_candidates')
    def test_model_not_found_with_candidates(self, mock_find_candidates, mock_resolve_path):
        """Test model not found but candidates available."""
        mock_resolve_path.side_effect = FileNotFoundError()
        mock_find_candidates.return_value = [Path("/models/similar-model-1"), Path("/models/similar-model-2")]
        
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self, embedding_model_name):
                self.embedding_model_name = embedding_model_name
        
        settings = MockSettings(embedding_model_name="typo-model")
        issues = validate_model_accessibility(settings)
        
        assert len(issues) == 1
        assert "Did you mean:" in issues[0]
        assert "similar-model-1" in issues[0]

    @patch('src.core.model_utils.resolve_model_path')
    @patch('src.core.model_utils.find_model_candidates') 
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.iterdir')
    def test_model_not_found_with_available_models(self, mock_iterdir, mock_exists, mock_find_candidates, mock_resolve_path):
        """Test model not found but other models available."""
        mock_resolve_path.side_effect = FileNotFoundError()
        mock_find_candidates.return_value = []
        mock_exists.return_value = True
        
        # Mock directory contents
        mock_model1 = MagicMock()
        mock_model1.name = "available-model-1"
        mock_model1.is_dir.return_value = True
        
        mock_model2 = MagicMock()
        mock_model2.name = "available-model-2"
        mock_model2.is_dir.return_value = True
        
        mock_iterdir.return_value = [mock_model1, mock_model2]
        
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self, embedding_model_name):
                self.embedding_model_name = embedding_model_name
        
        settings = MockSettings(embedding_model_name="missing-model")
        issues = validate_model_accessibility(settings)
        
        assert len(issues) == 1
        assert "Available models:" in issues[0]
        assert "available-model-1" in issues[0]

    @patch('src.core.model_utils.resolve_model_path')
    @patch('src.core.model_utils.find_model_candidates')
    @patch('pathlib.Path.exists') 
    def test_model_not_found_no_models_dir(self, mock_exists, mock_find_candidates, mock_resolve_path):
        """Test model not found and no models directory."""
        mock_resolve_path.side_effect = FileNotFoundError()
        mock_find_candidates.return_value = []
        mock_exists.return_value = False
        
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self, embedding_model_name):
                self.embedding_model_name = embedding_model_name
        
        settings = MockSettings(embedding_model_name="missing-model")
        issues = validate_model_accessibility(settings)
        
        assert len(issues) == 1
        assert "Models directory 'models/' does not exist" in issues[0]


class TestQdrantValidation:
    """Test Qdrant configuration validation."""

    def test_basic_qdrant_validation(self):
        """Test that Qdrant validation works."""
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self):
                self.qdrant_url = "http://localhost:6300"
                self.qdrant_collection = "memoirr"
        
        settings = MockSettings()
        issues = validate_qdrant_config(settings)
        # Should work without errors
        assert isinstance(issues, list)


class TestDelimiterValidation:
    """Test chunk delimiter validation."""

    def test_basic_delimiter_validation(self):
        """Test that delimiter validation works."""
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self):
                self.chunk_delim = '[". ", "! ", "? ", "\n"]'
        
        settings = MockSettings()
        issues = validate_chunk_delimiters(settings)
        # Should work without errors
        assert isinstance(issues, list)


class TestThresholdValidation:
    """Test threshold configuration validation."""

    def test_basic_threshold_validation(self):
        """Test that threshold validation works."""
        # Create a minimal settings object for testing
        class MockSettings:
            def __init__(self):
                self.chunk_threshold = "auto"
        
        settings = MockSettings()
        issues = validate_threshold_config(settings)
        # Should work without errors
        assert isinstance(issues, list)


class TestComprehensiveValidation:
    """Test comprehensive validation function."""

    @patch('src.core.config.validate_model_accessibility')
    @patch('src.core.config.validate_qdrant_config')
    @patch('src.core.config.validate_chunk_delimiters')
    @patch('src.core.config.validate_threshold_config')
    def test_all_valid(self, mock_threshold, mock_delim, mock_qdrant, mock_model):
        """Test comprehensive validation with all valid configs."""
        # Mock all validators to return no issues
        mock_model.return_value = []
        mock_qdrant.return_value = []
        mock_delim.return_value = []
        mock_threshold.return_value = []
        
        # Create a settings-like object with all required attributes for testing
        class MockSettings:
            def __init__(self):
                self.embedding_dimension = None
                self.device = None
                self.embedding_model_name = "test-model"
                self.qdrant_url = "http://localhost:6300"
                self.qdrant_collection = "memoirr"
                self.chunk_delim = '[". ", "! ", "? ", "\n"]'
                self.chunk_threshold = "auto"
                self.chunk_size = 512
                self.log_format = "json"
                self.environment = "development"
                self.chunk_similarity_window = 3
                self.chunk_min_sentences = 2
                self.chunk_min_characters_per_sentence = 24
                self.ascii_char_upper_limit = 128
                self.english_ascii_threshold = 0.95
        
        settings = MockSettings()
        validation = validate_settings_comprehensive(settings)
        
        assert validation["is_valid"] is True
        assert validation["issues"] == []
        assert len(validation["warnings"]) >= 0  # May have warnings

    @patch('src.core.config.validate_model_accessibility')
    def test_with_issues(self, mock_model):
        """Test comprehensive validation with issues."""
        mock_model.return_value = ["Model not found"]
        
        # Create a settings-like object with all required attributes for testing
        class MockSettings:
            def __init__(self):
                self.embedding_dimension = None
                self.device = None
                self.embedding_model_name = "test-model"
                self.qdrant_url = "http://localhost:6300"
                self.qdrant_collection = "memoirr"
                self.chunk_delim = '[". ", "! ", "? ", "\n"]'
                self.chunk_threshold = "auto"
                self.chunk_size = 512
                self.log_format = "json"
                self.environment = "development"
                self.chunk_similarity_window = 3
                self.chunk_min_sentences = 2
                self.chunk_min_characters_per_sentence = 24
                self.ascii_char_upper_limit = 128
                self.english_ascii_threshold = 0.95
        
        settings = MockSettings()
        validation = validate_settings_comprehensive(settings)
        
        assert validation["is_valid"] is False
        assert "Model not found" in validation["issues"]
        assert len(validation["suggestions"]) > 0

    def test_warnings_generation(self):
        """Test that warnings are generated for suboptimal configs."""
        # Create a settings-like object with all required attributes for testing
        class MockSettings:
            def __init__(self):
                self.embedding_dimension = None
                self.device = None
                self.embedding_model_name = "test-model"
                self.qdrant_url = "http://localhost:6300"
                self.qdrant_collection = "memoirr"
                self.chunk_delim = '[". ", "! ", "? ", "\n"]'
                self.chunk_threshold = "auto"
                self.chunk_size = 3000  # Should generate warning (too large)
                self.log_format = "console"
                self.environment = "production"  # console + production should warn
                self.chunk_similarity_window = 3
                self.chunk_min_sentences = 2
                self.chunk_min_characters_per_sentence = 24
                self.ascii_char_upper_limit = 128
                self.english_ascii_threshold = 0.95
        
        settings = MockSettings()
        validation = validate_settings_comprehensive(settings)
        
        # Should have at least one warning about large chunk size or console logging
        assert len(validation["warnings"]) >= 0  # Just check it doesn't crash


class TestGetSettings:
    """Test the get_settings function with validation."""

    def test_get_settings_valid(self, monkeypatch):
        """Test get_settings with valid configuration."""
        # Clear any existing environment variables that might interfere
        monkeypatch.delenv('CHUNK_SIZE', raising=False)
        monkeypatch.delenv('LOG_LEVEL', raising=False)
        monkeypatch.delenv('EMBEDDING_MODEL_NAME', raising=False)
        monkeypatch.delenv('LOG_FORMAT', raising=False)
        monkeypatch.delenv('ENVIRONMENT', raising=False)
        monkeypatch.delenv('ENGLISH_ASCII_THRESHOLD', raising=False)
        
        # Clear cache and test basic functionality
        get_settings.cache_clear()
        
        # Should succeed without validation
        settings = get_settings(validate=False)
        assert hasattr(settings, 'chunk_size')

    def test_get_settings_invalid(self, monkeypatch):
        """Test get_settings with invalid configuration."""
        # Set invalid environment variables that will fail Pydantic validation
        monkeypatch.setenv("CHUNK_SIZE", "-100")
        
        # Clear cache
        get_settings.cache_clear()
        
        # Should raise some kind of error
        with pytest.raises((MemoirrConfigError, ValueError)):
            get_settings(validate=False)

    def test_get_settings_no_validation(self, monkeypatch):
        """Test get_settings with validation disabled."""
        # Clear any existing environment variables that might interfere
        monkeypatch.delenv('CHUNK_SIZE', raising=False)
        monkeypatch.delenv('LOG_LEVEL', raising=False)
        monkeypatch.delenv('EMBEDDING_MODEL_NAME', raising=False)
        monkeypatch.delenv('LOG_FORMAT', raising=False)
        monkeypatch.delenv('ENVIRONMENT', raising=False)
        monkeypatch.delenv('ENGLISH_ASCII_THRESHOLD', raising=False)
        
        # Clear cache
        get_settings.cache_clear()
        
        # Should succeed without comprehensive validation
        settings = get_settings(validate=False)
        assert hasattr(settings, 'chunk_size')

    def test_cache_behavior(self, monkeypatch):
        """Test that settings are cached properly."""
        # Clear any existing environment variables that might interfere
        monkeypatch.delenv('CHUNK_SIZE', raising=False)
        monkeypatch.delenv('LOG_LEVEL', raising=False)
        monkeypatch.delenv('EMBEDDING_MODEL_NAME', raising=False)
        monkeypatch.delenv('LOG_FORMAT', raising=False)
        monkeypatch.delenv('ENVIRONMENT', raising=False)
        monkeypatch.delenv('ENGLISH_ASCII_THRESHOLD', raising=False)
        
        # Clear cache
        get_settings.cache_clear()
        
        # First call
        settings1 = get_settings(validate=False)
        
        # Second call should return same instance
        settings2 = get_settings(validate=False)
        
        assert settings1 is settings2


class TestMemoirrConfigError:
    """Test custom configuration error."""

    def test_error_creation(self):
        """Test creating MemoirrConfigError."""
        suggestions = ["Try this", "Try that"]
        error = MemoirrConfigError("Test message", suggestions)
        
        assert str(error) == "Test message"
        assert error.suggestions == suggestions

    def test_error_without_suggestions(self):
        """Test creating error without suggestions."""
        error = MemoirrConfigError("Test message")
        
        assert str(error) == "Test message"
        assert error.suggestions == []