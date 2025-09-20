# Components

This directory contains all the Haystack components that make up the Memoirr RAG pipeline. Each subdirectory represents a specific processing stage in the pipeline.

## Component Overview

### 1. Preprocessor
Handles the initial cleaning and processing of raw SRT subtitle files into structured JSONL format.

### 2. Chunker
Performs semantic chunking of preprocessed text to create meaningful segments for embedding and retrieval.

### 3. Embedder
Generates vector embeddings for text using sentence-transformers models.

### 4. Retriever
Searches for relevant documents in the Qdrant vector store based on query embeddings.

### 5. Generator
Uses LLMs (like Groq) to generate responses based on retrieved context.

### 6. Writer
Stores processed documents into the Qdrant vector store for future retrieval.

Each component follows the Haystack component pattern with:
- `@component` decorator
- `run()` method with defined input/output types
- Proper error handling and logging
- Configuration via environment variables