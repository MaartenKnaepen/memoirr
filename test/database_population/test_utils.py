"""Tests for database population utility functions."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from src.database_population.utilities.utils import (
    validate_directory,
    get_file_size_mb,
    format_processing_summary,
    estimate_processing_time,
    get_directory_stats
)


class TestValidateDirectory:
    """Test the validate_directory function."""
    
    def test_validate_directory_success(self):
        """Test validation of valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise any exception
            validate_directory(temp_dir)
    
    def test_validate_directory_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            validate_directory("")
    
    def test_validate_directory_nonexistent(self):
        """Test validation with nonexistent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            validate_directory("/nonexistent/directory")
    
    def test_validate_directory_not_directory(self):
        """Test validation when path is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                validate_directory(temp_file.name)
    
    @patch('os.access')
    def test_validate_directory_not_readable(self, mock_access):
        """Test validation when directory is not readable."""
        mock_access.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(PermissionError, match="Directory is not readable"):
                validate_directory(temp_dir)


class TestGetFileSizeMb:
    """Test the get_file_size_mb function."""
    
    def test_get_file_size_mb_success(self):
        """Test getting file size for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write some content
            content = "x" * 1024 * 1024  # 1 MB
            temp_file.write(content.encode('utf-8'))
            temp_file.flush()
            
            try:
                size_mb = get_file_size_mb(temp_file.name)
                assert abs(size_mb - 1.0) < 0.1  # Should be approximately 1 MB
            finally:
                os.unlink(temp_file.name)
    
    def test_get_file_size_mb_nonexistent(self):
        """Test getting file size for nonexistent file."""
        size_mb = get_file_size_mb("/nonexistent/file.txt")
        assert size_mb == 0.0
    
    def test_get_file_size_mb_empty_file(self):
        """Test getting file size for empty file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            size_mb = get_file_size_mb(temp_file.name)
            assert size_mb == 0.0


class TestFormatProcessingSummary:
    """Test the format_processing_summary function."""
    
    def test_format_processing_summary_success(self):
        """Test formatting successful processing summary."""
        summary = format_processing_summary(
            total_files=10,
            successful_files=8,
            failed_files=2,
            total_documents=150,
            processing_time_ms=5000.0
        )
        
        assert "Total SRT files found: 10" in summary
        assert "Successfully processed: 8 (80.0%)" in summary
        assert "Failed: 2" in summary
        assert "Total documents written: 150" in summary
        assert "Processing time: 5.00 seconds" in summary
        assert "Average time per file: 500.0 ms" in summary
        assert "Documents per second: 30.0" in summary
    
    def test_format_processing_summary_zero_files(self):
        """Test formatting summary with zero files."""
        summary = format_processing_summary(
            total_files=0,
            successful_files=0,
            failed_files=0,
            total_documents=0,
            processing_time_ms=1000.0
        )
        
        assert "Total SRT files found: 0" in summary
        assert "Successfully processed: 0 (0.0%)" in summary
        assert "Average time per file" not in summary  # Should not appear for 0 files
    
    def test_format_processing_summary_zero_time(self):
        """Test formatting summary with zero processing time."""
        summary = format_processing_summary(
            total_files=5,
            successful_files=5,
            failed_files=0,
            total_documents=50,
            processing_time_ms=0.0
        )
        
        assert "Total SRT files found: 5" in summary
        assert "Documents per second" not in summary  # Should not appear for 0 time


class TestEstimateProcessingTime:
    """Test the estimate_processing_time function."""
    
    def test_estimate_processing_time_seconds(self):
        """Test estimation for short processing time."""
        estimate = estimate_processing_time(file_count=5, avg_file_size_mb=0.1)
        assert "seconds" in estimate
        assert "~" in estimate
    
    def test_estimate_processing_time_minutes(self):
        """Test estimation for medium processing time."""
        estimate = estimate_processing_time(file_count=50, avg_file_size_mb=1.0)
        assert "minutes" in estimate
        assert "~" in estimate
    
    def test_estimate_processing_time_hours(self):
        """Test estimation for long processing time."""
        estimate = estimate_processing_time(file_count=1000, avg_file_size_mb=5.0)
        assert "hours" in estimate
        assert "~" in estimate
    
    def test_estimate_processing_time_zero_files(self):
        """Test estimation with zero files."""
        estimate = estimate_processing_time(file_count=0, avg_file_size_mb=1.0)
        assert "seconds" in estimate


class TestGetDirectoryStats:
    """Test the get_directory_stats function."""
    
    @patch('src.database_population.utilities.batch_processor.file_operations.find_srt_files')
    @patch('src.database_population.utilities.utils.get_file_size_mb')
    @patch('src.database_population.utilities.utils.validate_directory')
    def test_get_directory_stats_success(
        self, 
        mock_validate,
        mock_get_size,
        mock_find_files
    ):
        """Test successful directory statistics gathering."""
        # Mock file discovery
        mock_find_files.return_value = [
            "/test/file1.srt",
            "/test/file2.srt",
            "/test/file3.srt"
        ]
        
        # Mock file sizes
        mock_get_size.side_effect = [1.5, 2.0, 0.5]  # MB
        
        stats = get_directory_stats("/test/directory")
        
        assert stats["file_count"] == 3
        assert stats["total_size_mb"] == 4.0
        assert stats["avg_size_mb"] == 4.0 / 3
        assert stats["largest_file"]["path"] == "/test/file2.srt"
        assert stats["largest_file"]["size_mb"] == 2.0
        assert stats["smallest_file"]["path"] == "/test/file3.srt"
        assert stats["smallest_file"]["size_mb"] == 0.5
    
    @patch('src.database_population.utilities.batch_processor.file_operations.find_srt_files')
    @patch('src.database_population.utilities.utils.validate_directory')
    def test_get_directory_stats_empty_directory(self, mock_validate, mock_find_files):
        """Test statistics for empty directory."""
        mock_find_files.return_value = []
        
        stats = get_directory_stats("/test/empty")
        
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["avg_size_mb"] == 0.0
        assert stats["largest_file"] is None
        assert stats["smallest_file"] is None
    
    @patch('src.database_population.utilities.utils.validate_directory')
    def test_get_directory_stats_invalid_directory(self, mock_validate):
        """Test error handling for invalid directory."""
        mock_validate.side_effect = ValueError("Directory does not exist")
        
        stats = get_directory_stats("/invalid/directory")
        
        assert "error" in stats
        assert stats["file_count"] == 0
        assert "Directory does not exist" in stats["error"]
    
    @patch('src.database_population.utilities.batch_processor.file_operations.find_srt_files')
    @patch('src.database_population.utilities.utils.get_file_size_mb')
    @patch('src.database_population.utilities.utils.validate_directory')
    def test_get_directory_stats_single_file(
        self,
        mock_validate,
        mock_get_size,
        mock_find_files
    ):
        """Test statistics for directory with single file."""
        mock_find_files.return_value = ["/test/single.srt"]
        mock_get_size.return_value = 1.0
        
        stats = get_directory_stats("/test/directory")
        
        assert stats["file_count"] == 1
        assert stats["total_size_mb"] == 1.0
        assert stats["avg_size_mb"] == 1.0
        assert stats["largest_file"]["path"] == "/test/single.srt"
        assert stats["smallest_file"]["path"] == "/test/single.srt"
        assert stats["largest_file"]["size_mb"] == 1.0
        assert stats["smallest_file"]["size_mb"] == 1.0