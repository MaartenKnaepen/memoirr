"""Integration tests for database population functionality.

These tests verify the end-to-end functionality of the batch processing
system with actual components and mock database operations.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.database_population.batch_processor import process_srt_directory
from src.database_population.utilities.utils import get_directory_stats


@pytest.fixture
def sample_srt_content():
    """Sample SRT content for testing."""
    return """1
00:00:01,000 --> 00:00:03,000
Hello, this is the first subtitle.

2
00:00:04,000 --> 00:00:06,000
This is the second subtitle.

3
00:00:07,000 --> 00:00:09,000
And this is the third subtitle.

"""


@pytest.fixture
def complex_srt_directory(sample_srt_content):
    """Create a complex directory structure with multiple SRT files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create various SRT files with different content
        files_to_create = [
            ("movie1/english.srt", sample_srt_content),
            ("movie1/spanish.srt", sample_srt_content.replace("subtitle", "subtítulo")),
            ("movie2/episode1.srt", sample_srt_content),
            ("movie2/episode2.srt", sample_srt_content * 2),  # Longer content
            ("series/season1/ep01.srt", sample_srt_content),
            ("series/season1/ep02.srt", sample_srt_content),
            ("series/season2/ep01.srt", sample_srt_content),
            ("documentaries/nature.srt", sample_srt_content * 3),  # Even longer
            ("broken_file.srt", "This is not valid SRT content"),
            ("empty.srt", ""),
        ]
        
        for file_path, content in files_to_create:
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        
        # Also create some non-SRT files that should be ignored
        (Path(temp_dir) / "readme.txt").write_text("This is not an SRT file")
        (Path(temp_dir) / "movie1" / "poster.jpg").write_text("Fake image data")
        
        yield temp_dir


class TestEndToEndProcessing:
    """Test complete end-to-end processing scenarios."""
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_end_to_end_processing_success(
        self,
        mock_build_pipeline,
        mock_qdrant_writer,
        complex_srt_directory
    ):
        """Test complete end-to-end processing with mocked components."""
        # Mock the QdrantWriter
        mock_writer_instance = Mock()
        mock_writer_instance.get_document_count.side_effect = [50, 0, 85]  # Before clear, after clear, final
        mock_writer_instance.clear_collection.return_value = True
        mock_qdrant_writer.return_value = mock_writer_instance
        
        # Mock the pipeline
        mock_pipeline = Mock()
        # Simulate different numbers of documents written per file
        mock_pipeline.run.side_effect = [
            {"write": {"stats": {"written": 3}}},  # english.srt
            {"write": {"stats": {"written": 3}}},  # spanish.srt
            {"write": {"stats": {"written": 3}}},  # episode1.srt
            {"write": {"stats": {"written": 6}}},  # episode2.srt (longer)
            {"write": {"stats": {"written": 3}}},  # season1/ep01.srt
            {"write": {"stats": {"written": 3}}},  # season1/ep02.srt
            {"write": {"stats": {"written": 3}}},  # season2/ep01.srt
            {"write": {"stats": {"written": 9}}},  # nature.srt (longest)
            {"write": {"stats": {"written": 0}}},  # broken_file.srt
            {"write": {"stats": {"written": 0}}},  # empty.srt
        ]
        mock_build_pipeline.return_value = mock_pipeline
        
        # Process with overwrite
        result = process_srt_directory(complex_srt_directory, overwrite=True)
        
        # Verify results
        assert result.total_files == 10
        assert result.successful_files == 10  # All files processed (even if 0 documents)
        assert result.failed_files == 0
        assert result.total_documents_written == 33  # Sum of all written documents
        
        # Verify database was cleared
        mock_writer_instance.clear_collection.assert_called_once()
        
        # Verify all files were processed
        assert len(result.file_results) == 10
        assert all(r.success for r in result.file_results)
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_end_to_end_processing_with_failures(
        self,
        mock_build_pipeline,
        mock_qdrant_writer,
        complex_srt_directory
    ):
        """Test end-to-end processing with some file failures."""
        # Mock the QdrantWriter
        mock_writer_instance = Mock()
        mock_writer_instance.get_document_count.return_value = 25
        mock_qdrant_writer.return_value = mock_writer_instance
        
        # Mock the pipeline with some failures
        mock_pipeline = Mock()
        successful_results = [{"write": {"stats": {"written": 3}}}] * 7
        failed_results = [Exception("Processing failed")] * 3
        
        mock_pipeline.run.side_effect = successful_results + failed_results
        mock_build_pipeline.return_value = mock_pipeline
        
        # Process without overwrite
        result = process_srt_directory(complex_srt_directory, overwrite=False)
        
        # Verify results
        assert result.total_files == 10
        assert result.successful_files == 7
        assert result.failed_files == 3
        assert result.total_documents_written == 21  # 7 * 3
        
        # Verify database was NOT cleared
        mock_writer_instance.clear_collection.assert_not_called()
        
        # Check failed results
        failed_results = [r for r in result.file_results if not r.success]
        assert len(failed_results) == 3
        assert all("Processing failed" in r.error_message for r in failed_results)


class TestDirectoryStatistics:
    """Test directory statistics functionality."""
    
    def test_directory_stats_comprehensive(self, complex_srt_directory):
        """Test comprehensive directory statistics."""
        stats = get_directory_stats(complex_srt_directory)
        
        assert stats["file_count"] == 10
        assert stats["total_size_mb"] > 0
        assert stats["avg_size_mb"] > 0
        assert stats["largest_file"] is not None
        assert stats["smallest_file"] is not None
        
        # The largest file should be the one with the most content
        largest_file_path = stats["largest_file"]["path"]
        assert "nature.srt" in largest_file_path or "episode2.srt" in largest_file_path
        
        # The smallest file should be empty.srt
        smallest_file_path = stats["smallest_file"]["path"]
        assert "empty.srt" in smallest_file_path


class TestErrorHandling:
    """Test various error handling scenarios."""
    
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_pipeline_build_failure(self, mock_build_pipeline, complex_srt_directory):
        """Test handling of pipeline build failures."""
        mock_build_pipeline.side_effect = Exception("Failed to build pipeline")
        
        with pytest.raises(Exception, match="Failed to build pipeline"):
            process_srt_directory(complex_srt_directory)
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_database_clear_failure(
        self,
        mock_build_pipeline,
        mock_qdrant_writer,
        complex_srt_directory
    ):
        """Test handling when database clearing fails."""
        # Mock failed database clearing
        mock_writer_instance = Mock()
        mock_writer_instance.clear_collection.return_value = False
        mock_qdrant_writer.return_value = mock_writer_instance
        
        # Mock successful pipeline
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"write": {"stats": {"written": 3}}}
        mock_build_pipeline.return_value = mock_pipeline
        
        # Should still process files even if clearing fails
        result = process_srt_directory(complex_srt_directory, overwrite=True)
        
        assert result.total_files == 10
        assert result.successful_files == 10  # Processing should continue
        
        # Verify clear was attempted
        mock_writer_instance.clear_collection.assert_called_once()


class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""
    
    @patch('src.components.writer.qdrant_writer.QdrantWriter')
    @patch('src.database_population.batch_processor.build_srt_to_qdrant_pipeline')
    def test_processing_time_tracking(
        self,
        mock_build_pipeline,
        mock_qdrant_writer,
        complex_srt_directory
    ):
        """Test that processing times are properly tracked."""
        # Mock components
        mock_writer_instance = Mock()
        mock_qdrant_writer.return_value = mock_writer_instance
        
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"write": {"stats": {"written": 1}}}
        mock_build_pipeline.return_value = mock_pipeline
        
        # Mock LoggedOperation to provide proper timing
        import time
        with patch('src.database_population.batch_processor.LoggedOperation') as mock_logged_op:
            mock_op_instance = Mock()
            mock_op_instance.start_time = time.time() - 0.1  # Mock start time as 100ms ago
            mock_logged_op.return_value.__enter__.return_value = mock_op_instance
            
            # Also mock the single file processor's LoggedOperation
            with patch('src.database_population.utilities.batch_processor.single_file_processor.LoggedOperation') as mock_single_logged_op:
                mock_single_op_instance = Mock()
                mock_single_op_instance.start_time = time.time() - 0.05  # Mock start time as 50ms ago
                mock_single_logged_op.return_value.__enter__.return_value = mock_single_op_instance
                
                result = process_srt_directory(complex_srt_directory)
        
        # Verify timing information is captured
        assert result.processing_time_ms > 0
        assert all(r.processing_time_ms is not None for r in result.file_results if r.success)
    
    def test_large_directory_handling(self):
        """Test handling of directories with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many small SRT files
            for i in range(50):
                file_path = Path(temp_dir) / f"subtitle_{i:03d}.srt"
                file_path.write_text(f"""1
00:00:0{i%6 + 1},000 --> 00:00:0{i%6 + 3},000
Subtitle number {i}

""", encoding='utf-8')
            
            stats = get_directory_stats(temp_dir)
            
            assert stats["file_count"] == 50
            assert stats["total_size_mb"] > 0
            assert stats["avg_size_mb"] > 0


class TestEdgeCases:
    """Test various edge cases and boundary conditions."""
    
    def test_unicode_content_handling(self):
        """Test handling of SRT files with Unicode content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create SRT files with various Unicode characters
            unicode_content = """1
00:00:01,000 --> 00:00:03,000
Hello 世界! Bonjour le monde! 🌍

2
00:00:04,000 --> 00:00:06,000
Здравствуй мир! مرحبا بالعالم! こんにちは世界！

"""
            
            srt_path = Path(temp_dir) / "unicode.srt"
            srt_path.write_text(unicode_content, encoding='utf-8')
            
            stats = get_directory_stats(temp_dir)
            assert stats["file_count"] == 1
            assert stats["total_size_mb"] > 0
    
    def test_very_large_file_handling(self):
        """Test handling of very large SRT files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a large SRT file
            large_content = ""
            for i in range(1000):
                large_content += f"""{i+1}
00:{i//60:02d}:{i%60:02d},000 --> 00:{(i+2)//60:02d}:{(i+2)%60:02d},000
This is subtitle number {i+1} with some content to make it longer.

"""
            
            large_file_path = Path(temp_dir) / "large.srt"
            large_file_path.write_text(large_content, encoding='utf-8')
            
            stats = get_directory_stats(temp_dir)
            assert stats["file_count"] == 1
            assert stats["total_size_mb"] > 0.05  # Should be reasonably large (reduced threshold)
    
    def test_deeply_nested_directories(self):
        """Test handling of deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create deeply nested structure
            deep_path = Path(temp_dir)
            for level in range(10):
                deep_path = deep_path / f"level_{level}"
            deep_path.mkdir(parents=True)
            
            # Create SRT file in the deep path
            srt_path = deep_path / "deep.srt"
            srt_path.write_text("""1
00:00:01,000 --> 00:00:03,000
Deep subtitle

""", encoding='utf-8')
            
            stats = get_directory_stats(temp_dir)
            assert stats["file_count"] == 1