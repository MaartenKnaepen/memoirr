# Pipelines

This directory contains the Haystack pipeline definitions that compose the components into end-to-end workflows.

## Pipeline Files

### srt_to_qdrant.py
The indexing pipeline that processes SRT files and stores them in Qdrant:
1. Preprocesses SRT files into clean JSONL
2. Semantically chunks the text
3. Generates embeddings for each chunk
4. Writes the embedded chunks to Qdrant vector store

### rag_pipeline.py
The Retrieval-Augmented Generation (RAG) pipeline for query processing:
1. Embeds the user query
2. Retrieves relevant documents from Qdrant
3. Generates contextual responses using an LLM

## Utilities

The utilities subdirectory contains helper functions for pipeline construction, connection management, and execution coordination.