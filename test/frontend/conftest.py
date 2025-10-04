"""Frontend test configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for all frontend tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@pytest.fixture
def mock_model_path():
    """Provide a mock model path for tests."""
    return Path("/fake/model/path")

@pytest.fixture
def mock_settings():
    """Provide mock settings for tests."""
    from unittest.mock import MagicMock
    
    settings = MagicMock()
    settings.embedding_model_name = "test-model"
    settings.embedding_dimension = 512
    settings.device = None
    settings.embedding_dimension_fallback = 1024
    
    return settings