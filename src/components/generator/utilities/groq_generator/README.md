# Groq Generator Utilities

This directory contains the utility functions that implement the Groq LLM generation pipeline. These functions are orchestrated by the GroqGenerator component.

## Modules

### orchestrate_generation.py
The main orchestration function that manages the complete generation pipeline:
- Builds prompts from query and retrieved context
- Calls the Groq API with proper error handling
- Processes and formats the LLM responses

### prompt_builder.py
Constructs system and user prompts for the LLM using Jinja2 templates:
- Builds RAG prompts with document context
- Formats system instructions based on query type
- Truncates context to fit within model limits

### response_processor.py
Processes raw LLM responses into structured formats:
- Extracts reply text from API responses
- Parses metadata like sources, timing, and speakers
- Formats replies for display with proper attribution