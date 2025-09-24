"""Database operation utilities for batch processing."""

from src.core.logging_config import get_logger


def clear_qdrant_database() -> bool:
    """Clear the Qdrant database by removing all documents from the collection.
    
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        # Import locally to avoid circular imports
        from src.components.writer.qdrant_writer import QdrantWriter
        
        # Create a temporary writer instance to access the collection management
        writer = QdrantWriter()
        
        # Get document count before clearing
        doc_count_before = writer.get_document_count()
        logger.info(
            "Clearing Qdrant database",
            documents_before=doc_count_before,
            component="batch_processor"
        )
        
        # Clear the collection
        success = writer.clear_collection()
        
        if success:
            # Verify clearing was successful
            doc_count_after = writer.get_document_count()
            logger.info(
                "Database cleared successfully",
                documents_before=doc_count_before,
                documents_after=doc_count_after,
                component="batch_processor"
            )
        
        return success
        
    except Exception as e:
        logger.error(
            "Failed to clear Qdrant database",
            error=str(e),
            error_type=type(e).__name__,
            component="batch_processor"
        )
        return False