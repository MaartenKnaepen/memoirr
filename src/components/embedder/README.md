# Embedder

This directory contains the TextEmbedder component, which generates vector embeddings for text using local sentence-transformers models.

## Files

### text_embedder.py
The main Haystack component that:
- Loads a local sentence-transformers model
- Generates embeddings for input text
- Handles batch processing for efficiency
- Provides error handling and fallback mechanisms

The embedder uses models stored in the local `models/` directory and is configured through environment variables for model name, device, and dimension settings.