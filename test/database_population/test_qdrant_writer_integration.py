"""Tests for QdrantWriter integration with database population functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.components.writer.qdrant_writer import QdrantWriter


class TestQdrantWriterDatabaseOperations:
    """Test QdrantWriter database clearing and counting operations."""
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_clear_collection_success(self, mock_get_settings, mock_qdrant_store):
        """Test successful collection clearing."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        # Create writer and test clearing
        writer = QdrantWriter()
        result = writer.clear_collection()
        
        assert result is True
        mock_store_instance.delete_documents.assert_called_once()
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_clear_collection_failure(self, mock_get_settings, mock_qdrant_store):
        """Test collection clearing failure."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store that fails
        mock_store_instance = Mock()
        mock_store_instance.delete_documents.side_effect = Exception("Database error")
        mock_qdrant_store.return_value = mock_store_instance
        
        # Create writer and test clearing
        writer = QdrantWriter()
        result = writer.clear_collection()
        
        assert result is False
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_get_document_count_success(self, mock_get_settings, mock_qdrant_store):
        """Test successful document count retrieval."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store
        mock_store_instance = Mock()
        mock_store_instance.count_documents.return_value = 42
        mock_qdrant_store.return_value = mock_store_instance
        
        # Create writer and test count
        writer = QdrantWriter()
        count = writer.get_document_count()
        
        assert count == 42
        mock_store_instance.count_documents.assert_called_once()
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_get_document_count_failure(self, mock_get_settings, mock_qdrant_store):
        """Test document count retrieval failure."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store that fails
        mock_store_instance = Mock()
        mock_store_instance.count_documents.side_effect = Exception("Database error")
        mock_qdrant_store.return_value = mock_store_instance
        
        # Create writer and test count
        writer = QdrantWriter()
        count = writer.get_document_count()
        
        assert count == -1
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_write_documents_with_clearing_workflow(self, mock_get_settings, mock_qdrant_store):
        """Test the complete workflow: clear, count, write, count again."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store
        mock_store_instance = Mock()
        # Simulate count before clear: 50, after clear: 0, after write: 3
        mock_store_instance.count_documents.side_effect = [50, 0, 3]
        mock_qdrant_store.return_value = mock_store_instance
        
        # Create writer and simulate workflow
        writer = QdrantWriter()
        
        # 1. Get initial count
        initial_count = writer.get_document_count()
        assert initial_count == 50
        
        # 2. Clear collection
        clear_result = writer.clear_collection()
        assert clear_result is True
        
        # 3. Verify cleared
        cleared_count = writer.get_document_count()
        assert cleared_count == 0
        
        # 4. Write some documents
        test_documents = [
            {
                "content": "Test content 1",
                "embedding": [0.1, 0.2, 0.3],
                "meta": {"source": "test1.srt"}
            },
            {
                "content": "Test content 2", 
                "embedding": [0.4, 0.5, 0.6],
                "meta": {"source": "test2.srt"}
            },
            {
                "content": "Test content 3",
                "embedding": [0.7, 0.8, 0.9],
                "meta": {"source": "test3.srt"}
            }
        ]
        
        write_result = writer.run(test_documents)
        assert write_result["stats"]["written"] == 3
        
        # 5. Get final count to trigger the third call
        final_count = writer.get_document_count()
        assert final_count == 3
        
        # Verify all expected calls were made
        mock_store_instance.delete_documents.assert_called_once()
        mock_store_instance.write_documents.assert_called_once()
        assert mock_store_instance.count_documents.call_count == 3


class TestQdrantWriterErrorScenarios:
    """Test various error scenarios with QdrantWriter."""
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_initialization_failure(self, mock_get_settings, mock_qdrant_store):
        """Test QdrantWriter initialization failure."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store initialization failure
        mock_qdrant_store.side_effect = Exception("Failed to connect to Qdrant")
        
        with pytest.raises(Exception, match="Failed to connect to Qdrant"):
            QdrantWriter()
    
    @patch('src.components.writer.qdrant_writer.QdrantDocumentStore')
    @patch('src.components.writer.qdrant_writer.get_settings')
    def test_partial_clearing_failure(self, mock_get_settings, mock_qdrant_store):
        """Test scenario where clearing partially fails."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_collection = "test_collection"
        mock_settings.qdrant_recreate_index = False
        mock_settings.qdrant_return_embedding = False
        mock_settings.qdrant_wait_result = True
        mock_get_settings.return_value = mock_settings
        
        # Mock document store
        mock_store_instance = Mock()
        # Clearing succeeds but count afterwards fails
        mock_store_instance.count_documents.side_effect = [100, Exception("Count failed")]
        mock_qdrant_store.return_value = mock_store_instance
        
        writer = QdrantWriter()
        
        # Initial count works
        initial_count = writer.get_document_count()
        assert initial_count == 100
        
        # Clear succeeds
        clear_result = writer.clear_collection()
        assert clear_result is True
        
        # But subsequent count fails
        final_count = writer.get_document_count()
        assert final_count == -1