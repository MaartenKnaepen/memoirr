# Memoirr Frontend

Simple chat interface for the Memoirr RAG pipeline using Gradio.

## Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   uv sync
   ```

2. **Launch the chat interface**:
   ```bash
   python src/frontend/gradio_app.py
   ```

3. **Open your browser** to `http://localhost:7860`

## Features

- 💬 **Chat Interface**: Ask questions about subtitle content
- 📽️ **Source Attribution**: See which subtitle segments were used
- 🎯 **Example Queries**: Pre-built questions to get started
- 🔄 **Real-time Responses**: Powered by your existing RAG pipeline

## Example Questions

- "What did Tony Stark say about technology?"
- "Who are the main characters in this story?"
- "What happened in the first 10 minutes?"
- "What was the most emotional scene?"
- "Can you summarize the plot?"

## Configuration

The interface uses your existing configuration from `.env` files:
- Qdrant connection settings
- Groq API key
- Embedding model settings
- RAG pipeline parameters

## Troubleshooting

**Interface won't start:**
- Check that your RAG pipeline is properly configured
- Ensure Qdrant is running and accessible
- Verify your Groq API key is set

**No responses:**
- Verify your vector database is populated with SRT data
- Check the logs for detailed error messages

**Poor responses:**
- Try rephrasing your question
- Use more specific queries
- Check if your SRT data covers the topic you're asking about