# Memoirr Documentation

This directory contains comprehensive documentation for the Memoirr project, covering architecture, development guidelines, and operational procedures.

## Documentation Files

### [Database Population Strategies](database_population_strategies.md)
Strategies and approaches for populating the Qdrant database with processed media content. Covers batch processing, parallelization, and monitoring approaches.

### [Future Development Roadmap](future_development.md)
Detailed roadmap outlining planned features, architectural improvements, and strategic enhancements categorized by impact and effort. Includes short-term quality of life improvements, medium-term capability upgrades, and long-term platform vision.

### [Logging Guide](logging_guide.md)
Comprehensive guide to the structured logging system designed for observability with Grafana, Loki, and Prometheus. Includes quick start instructions, configuration options, logging patterns, and integration with the observability stack.

### [Python Coding Standards & Architectural Guide](qwen.md)
Mandatory style guides, architectural principles, and best practices for Memoirr development. Covers code style, architectural principles, development workflow, and Haystack component guidelines.

## Overview

Memoirr is a Retrieval-Augmented Generation (RAG) system for personal media libraries that enables semantic search and intelligent querying of movie and TV show content through subtitle processing and vector embeddings.

The system consists of two main pipelines:
1. **Indexing Pipeline**: Processes SRT files and stores embeddings in Qdrant
2. **Retrieval Pipeline**: Answers queries using retrieved context and LLM generation

## Key Components

- **SRT Preprocessor**: Cleans and structures subtitle content
- **Semantic Chunker**: Creates meaningful content segments
- **Text Embedder**: Generates vector embeddings using local models
- **Qdrant Storage**: Vector database for efficient retrieval
- **Groq Generator**: LLM-based response generation