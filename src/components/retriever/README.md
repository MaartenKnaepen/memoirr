# Retriever

This directory contains the QdrantRetriever component, which is responsible for retrieving relevant documents from the Qdrant vector store based on query embeddings.

## Files

### qdrant_retriever.py
The main Haystack component that:
- Accepts a text query
- Embeds the query using the configured embedding model
- Searches the Qdrant vector store for similar documents
- Returns ranked documents with similarity scores and metadata

## Utilities

Helper functions for orchestrating the retrieval process, including:
- Query embedding
- Document search and filtering
- Result ranking and metadata enhancement