# Writer

This directory contains the QdrantWriter component, which stores processed documents into the Qdrant vector store.

## Files

### qdrant_writer.py
The main Haystack component that:
- Accepts lists of documents with embeddings
- Writes them to the configured Qdrant collection
- Handles connection management and error recovery
- Provides feedback on write operations

This component is used in the indexing pipeline to persist processed and embedded document chunks for future retrieval.