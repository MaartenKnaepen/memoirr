# Qdrant Retriever Utilities

This directory contains the orchestration function for the Qdrant retrieval pipeline.

## Modules

### orchestrate_retrieval.py
The main orchestration function that implements the complete retrieval pipeline:
- Embeds the query text using the configured embedding model
- Searches Qdrant for similar document embeddings using QdrantEmbeddingRetriever
- Filters results by score threshold
- Returns ranked documents with metadata

This function is called by the QdrantRetriever component to handle the core retrieval logic.