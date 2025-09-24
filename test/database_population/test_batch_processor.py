"""Tests for the batch processor functionality."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.database_population.batch_processor import process_srt_directory
from src.database_population.utilities.batch_processor.types import (
    ProcessingResult,
    BatchProcessingResult
)
from src.database_population.utilities.batch_processor.file_operations import find_srt_files
from src.database_population.utilities.batch_processor.single_file_processor import process_single_srt_file
from src.database_population.utilities.batch_processor.database_operations import clear_qdrant_database


@pytest.fixture
def temp_srt_directory():
    """Create a temporary directory with sample SRT files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some sample SRT content
        sample_srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello, this is a test subtitle.

2
00:00:04,000 --> 00:00:06,000
This is another test subtitle.

"""
        
        # Create directory structure with SRT files
        srt_files = [
            "test1.srt",
            "subdir/test2.srt",
            "subdir/nested/test3.srt",
            "invalid.txt",  # Should be ignored
        ]
        
        for file_path in srt_files:
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.endswith('.srt'):
                full_path.write_text(sample_srt_content, encoding='utf-8')
            else:
                full_path.write_text("Not an SRT file", encoding='utf-8')
        
        yield temp_dir


class TestFindSrtFiles:
    """Test the find_srt_files function."""
    
    def test_find_srt_files_success(self, temp_srt_directory):
        """Test successful SRT file discovery."""
        files = find_srt_files(temp_srt_directory)
        
        assert len(files) == 3
        assert all(f.endswith('.srt') for f in files)
        assert all(os.path.exists(f) for f in files)
    
    def test_find_srt_files_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            find_srt_files("/nonexistent/directory")
    
    def test_find_srt_files_not_directory(self, temp_srt_directory):
        """Test error handling when path is not a directory."""
        file_path = os.path.join(temp_srt_directory, "test1.srt")
        with pytest.raises(ValueError, match="Path is not a directory"):
            find_srt_files(file_path)
    
    def test_find_srt_files_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = find_srt_files(temp_dir)
            assert files == []


class TestProcessSingleSrtFile:
    """Test the process_single_srt_file function."""
    
    @patch('src.database_population.utilities.batch_processor.single_file_processor.open')
    def test_process_single_srt_file_success(self, mock_open):
        """Test successful processing of a single SRT file."""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "Sample SRT content"
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "write": {"stats": {"written": 5}}
        }
        
        # Mock LoggedOperation to avoid duration_ms issues
        with patch('src.database_population.utilities.batch_processor.single_file_processor.LoggedOperation') as mock_logged_op:
            mock_op_instance = Mock()
            mock_op_instance.start_time = 1000.0  # Mock start time as a float
            mock_logged_op.return_value.__enter__.return_value = mock_op_instance
            
            result = process_single_srt_file("/test/file.srt", mock_pipeline)
            
            assert result.success is True
            assert result.file_path == "/test/file.srt"
            assert result.documents_written == 5
            assert result.error_message is None
            
            # Verify pipeline was called correctly
            mock_pipeline.run.assert_called_once_with({"pre": {"srt_text": "Sample SRT content"}})
    
    @patch('src.database_population.utilities.batch_processor.single_file_processor.open')
    def test_process_single_srt_file_read_error(self, mock_open):
        """Test handling of file read errors."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        mock_pipeline = Mock()
        
        result = process_single_srt_file("/nonexistent/file.srt", mock_pipeline)
        
        assert result.success is False
        assert result.file_path == "/nonexistent/file.srt"
        assert result.documents_written == 0
        assert "File not found" in result.error_message
    
    @patch('src.database_population.utilities.batch_processor.single_file_processor.open')
    def test_process_single_srt_file_pipeline_error(self, mock_open):
        """Test handling of pipeline errors."""
        mock_open.return_value.__enter__.return_value.read.return_value = "Sample SRT content"
        
        # Mock pipeline that raises an error
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = Exception("Pipeline error")
        
        result = process_single_srt_file("/test/file.srt", mock_pipeline)
        
        assert result.success is False
        assert result.file_path == "/test/file.srt"
        assert result.documents_written == 0
        assert "Pipeline error" in result.error_message


class TestClearQdrantDatabase:
    """Test the clear_qdrant_database function."""
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    def test_clear_qdrant_database_success(self, mock_writer_class):
        """Test successful database clearing."""
        mock_writer = Mock()
        mock_writer.get_document_count.side_effect = [100, 0]  # Before and after clearing
        mock_writer.clear_collection.return_value = True
        mock_writer_class.return_value = mock_writer
        
        result = clear_qdrant_database()
        
        assert result is True
        mock_writer.clear_collection.assert_called_once()
        assert mock_writer.get_document_count.call_count == 2
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    def test_clear_qdrant_database_failure(self, mock_writer_class):
        """Test database clearing failure."""
        mock_writer = Mock()
        mock_writer.clear_collection.return_value = False
        mock_writer_class.return_value = mock_writer
        
        result = clear_qdrant_database()
        
        assert result is False
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    def test_clear_qdrant_database_exception(self, mock_writer_class):
        """Test exception handling during database clearing."""
        mock_writer_class.side_effect = Exception("Connection error")
        
        result = clear_qdrant_database()
        
        assert result is False


class TestProcessSrtDirectory:
    """Test the process_srt_directory function."""
    
    @patch('src.database_population.utilities.batch_processor.database_operations.clear_qdrant_database')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    @patch('src.database_population.batch_processor.process_single_srt_file')
    def test_process_srt_directory_success(
        self, 
        mock_process_single,
        mock_build_pipeline,
        mock_clear_db,
        temp_srt_directory
    ):
        """Test successful directory processing."""
        # Mock pipeline completely to avoid any component instantiation
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"write": {"stats": {"written": 33}}}
        mock_build_pipeline.return_value = mock_pipeline
        
        # Mock successful file processing
        mock_process_single.side_effect = [
            ProcessingResult("/file1.srt", True, 10),
            ProcessingResult("/file2.srt", True, 15),
            ProcessingResult("/file3.srt", True, 8),
        ]
        
        # Mock LoggedOperation for the batch operation
        with patch('src.database_population.batch_processor.LoggedOperation') as mock_logged_op:
            mock_op_instance = Mock()
            mock_op_instance.start_time = 1000.0  # Mock start time as a float
            mock_logged_op.return_value.__enter__.return_value = mock_op_instance
            
            result = process_srt_directory(temp_srt_directory, overwrite=False)
        
        assert isinstance(result, BatchProcessingResult)
        assert result.total_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_documents_written == 33  # 10 + 15 + 8
        
        # Verify clear_database was not called (overwrite=False)
        mock_clear_db.assert_not_called()
    
    @patch('src.database_population.batch_processor.clear_qdrant_database')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    @patch('src.database_population.batch_processor.process_single_srt_file')
    def test_process_srt_directory_with_overwrite(
        self,
        mock_process_single,
        mock_build_pipeline,
        mock_clear_db,
        temp_srt_directory
    ):
        """Test directory processing with database overwrite."""
        # Mock successful database clearing
        mock_clear_db.return_value = True
        
        # Mock pipeline completely to avoid any component instantiation
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"write": {"stats": {"written": 15}}}
        mock_build_pipeline.return_value = mock_pipeline
        
        # Mock successful file processing
        mock_process_single.side_effect = [
            ProcessingResult("/file1.srt", True, 5),
            ProcessingResult("/file2.srt", True, 7),
            ProcessingResult("/file3.srt", True, 3),
        ]
        
        # Mock LoggedOperation for the batch operation
        with patch('src.database_population.batch_processor.LoggedOperation') as mock_logged_op:
            mock_op_instance = Mock()
            mock_op_instance.start_time = 1000.0  # Mock start time as a float
            mock_logged_op.return_value.__enter__.return_value = mock_op_instance
            
            result = process_srt_directory(temp_srt_directory, overwrite=True)
        
        assert result.successful_files == 3
        assert result.total_documents_written == 15
        
        # Verify clear_database was called
        mock_clear_db.assert_called_once()
    
    @patch('src.database_population.batch_processor.clear_qdrant_database')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    @patch('src.database_population.batch_processor.process_single_srt_file')
    def test_process_srt_directory_mixed_results(
        self,
        mock_process_single,
        mock_build_pipeline,
        mock_clear_db,
        temp_srt_directory
    ):
        """Test directory processing with mixed success/failure results."""
        # Mock pipeline completely to avoid any component instantiation
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"write": {"stats": {"written": 15}}}
        mock_build_pipeline.return_value = mock_pipeline
        
        # Mock mixed results
        mock_process_single.side_effect = [
            ProcessingResult("/file1.srt", True, 10),
            ProcessingResult("/file2.srt", False, 0, "Processing error"),
            ProcessingResult("/file3.srt", True, 5),
        ]
        
        # Mock LoggedOperation for the batch operation
        with patch('src.database_population.batch_processor.LoggedOperation') as mock_logged_op:
            mock_op_instance = Mock()
            mock_op_instance.start_time = 1000.0  # Mock start time as a float
            mock_logged_op.return_value.__enter__.return_value = mock_op_instance
            
            result = process_srt_directory(temp_srt_directory, overwrite=False)
        
        assert result.total_files == 3
        assert result.successful_files == 2
        assert result.failed_files == 1
        assert result.total_documents_written == 15  # Only successful files count
        
        # Check individual results
        failed_results = [r for r in result.file_results if not r.success]
        assert len(failed_results) == 1
        assert "Processing error" in failed_results[0].error_message
    
    def test_process_srt_directory_invalid_path(self):
        """Test error handling for invalid directory path."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            process_srt_directory("/nonexistent/directory")
    
    def test_process_srt_directory_empty(self):
        """Test processing empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_srt_directory(temp_dir)
            
            assert result.total_files == 0
            assert result.successful_files == 0
            assert result.failed_files == 0
            assert result.total_documents_written == 0
            assert result.file_results == []
    
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_process_srt_directory_pipeline_build_error(
        self,
        mock_build_pipeline,
        temp_srt_directory
    ):
        """Test error handling when pipeline build fails."""
        mock_build_pipeline.side_effect = Exception("Pipeline build error")
        
        with pytest.raises(Exception, match=r"Pipeline build error"):
            process_srt_directory(temp_srt_directory)


class TestProcessingResult:
    """Test the ProcessingResult dataclass."""
    
    def test_processing_result_success(self):
        """Test ProcessingResult for successful processing."""
        result = ProcessingResult(
            file_path="/test/file.srt",
            success=True,
            documents_written=10,
            processing_time_ms=1500.0
        )
        
        assert result.file_path == "/test/file.srt"
        assert result.success is True
        assert result.documents_written == 10
        assert result.error_message is None
        assert result.processing_time_ms == 1500.0
    
    def test_processing_result_failure(self):
        """Test ProcessingResult for failed processing."""
        result = ProcessingResult(
            file_path="/test/file.srt",
            success=False,
            error_message="File not found"
        )
        
        assert result.file_path == "/test/file.srt"
        assert result.success is False
        assert result.documents_written == 0  # Default value
        assert result.error_message == "File not found"
        assert result.processing_time_ms is None


class TestBatchProcessingResult:
    """Test the BatchProcessingResult dataclass."""
    
    def test_batch_processing_result(self):
        """Test BatchProcessingResult construction."""
        file_results = [
            ProcessingResult("/file1.srt", True, 10),
            ProcessingResult("/file2.srt", False, 0, "Error"),
        ]
        
        result = BatchProcessingResult(
            total_files=2,
            successful_files=1,
            failed_files=1,
            total_documents_written=10,
            file_results=file_results,
            processing_time_ms=5000.0
        )
        
        assert result.total_files == 2
        assert result.successful_files == 1
        assert result.failed_files == 1
        assert result.total_documents_written == 10
        assert len(result.file_results) == 2
        assert result.processing_time_ms == 5000.0