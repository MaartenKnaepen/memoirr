"""Pytest configuration and fixtures for database population tests."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_collection"
    settings.qdrant_recreate_index = False
    settings.qdrant_return_embedding = False
    settings.qdrant_wait_result = True
    settings.embedding_dimension = 384
    return settings


@pytest.fixture
def sample_srt_content():
    """Standard sample SRT content for testing."""
    return """1
00:00:01,000 --> 00:00:03,000
This is the first subtitle line.

2
00:00:04,000 --> 00:00:06,000
This is the second subtitle line.

3
00:00:07,000 --> 00:00:09,000
And this is the final subtitle line.

"""


@pytest.fixture
def temp_srt_file(sample_srt_content):
    """Create a temporary SRT file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(sample_srt_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_directory_with_srt_files(sample_srt_content):
    """Create a temporary directory with multiple SRT files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main directory files
        files = [
            ("subtitle1.srt", sample_srt_content),
            ("subtitle2.srt", sample_srt_content * 2),
            ("subtitle3.srt", sample_srt_content),
        ]
        
        # Create subdirectory files
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        files.extend([
            ("subdir/sub_subtitle1.srt", sample_srt_content),
            ("subdir/sub_subtitle2.srt", sample_srt_content),
        ])
        
        # Create nested subdirectory files
        nested_subdir = subdir / "nested"
        nested_subdir.mkdir()
        files.append(("subdir/nested/nested_subtitle.srt", sample_srt_content))
        
        # Write all files
        for file_path, content in files:
            full_path = Path(temp_dir) / file_path
            full_path.write_text(content, encoding='utf-8')
        
        # Also create some non-SRT files that should be ignored
        (Path(temp_dir) / "readme.txt").write_text("Not an SRT file")
        (Path(temp_dir) / "subdir" / "info.md").write_text("Also not an SRT file")
        
        yield temp_dir


@pytest.fixture
def mock_pipeline_success():
    """Mock pipeline that always succeeds."""
    pipeline = Mock()
    pipeline.run.return_value = {
        "write": {"stats": {"written": 3}}
    }
    return pipeline


@pytest.fixture
def mock_pipeline_failure():
    """Mock pipeline that always fails."""
    pipeline = Mock()
    pipeline.run.side_effect = Exception("Pipeline processing failed")
    return pipeline


@pytest.fixture
def mock_qdrant_writer_success():
    """Mock QdrantWriter that always succeeds."""
    with patch('src.components.writer.qdrant_writer.QdrantWriter') as mock_class:
        mock_instance = Mock()
        mock_instance.clear_collection.return_value = True
        mock_instance.get_document_count.return_value = 0
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_writer_failure():
    """Mock QdrantWriter that fails operations."""
    with patch('src.components.writer.qdrant_writer.QdrantWriter') as mock_class:
        mock_instance = Mock()
        mock_instance.clear_collection.return_value = False
        mock_instance.get_document_count.side_effect = Exception("Database connection failed")
        mock_class.return_value = mock_instance
        yield mock_instance


# Test data constants
SAMPLE_PROCESSING_RESULT_SUCCESS = {
    "file_path": "/test/sample.srt",
    "success": True,
    "documents_written": 5,
    "processing_time_ms": 1500.0
}

SAMPLE_PROCESSING_RESULT_FAILURE = {
    "file_path": "/test/failed.srt", 
    "success": False,
    "documents_written": 0,
    "error_message": "Processing failed",
    "processing_time_ms": None
}

SAMPLE_BATCH_RESULT = {
    "total_files": 10,
    "successful_files": 8,
    "failed_files": 2,
    "total_documents_written": 45,
    "processing_time_ms": 15000.0
}