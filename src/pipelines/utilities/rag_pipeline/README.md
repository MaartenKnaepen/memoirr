# RAG Pipeline Utilities

This directory contains utility functions for the RAG (Retrieval-Augmented Generation) pipeline.

## Modules

### orchestrate_rag_query.py
The main orchestration function that manages the complete RAG query pipeline:
- Processes user queries through the retrieval and generation components
- Handles query routing and parameter management
- Coordinates the flow between retrieval and generation phases
- Manages error handling and fallback strategies