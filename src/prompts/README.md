# Prompts

This directory contains the Jinja2 templates and utilities for generating system prompts used by the LLM generator component.

## Files

### template_loader.py
Utility functions for loading and rendering Jinja2 prompt templates with context variables.

### default_system.j2
The default system prompt template that provides the LLM with its role, instructions, and context format. This template supports different analysis types:
- General analysis
- Character analysis
- Quote finding
- Timeline creation

The template uses Jinja2 syntax to dynamically insert:
- Document context
- Query instructions
- Formatting guidelines
- Example responses