"""Tests for prompt building utilities.

Tests prompt construction, context formatting, and subtitle-specific optimizations.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document
from src.components.generator.utilities.groq_generator.prompt_builder import (
    build_rag_prompt,
    build_system_prompt,
    truncate_context_to_limit,
    _build_query_only_prompt,
    _build_contextualized_prompt,
    _format_document_context,
)


class TestPromptBuilder:
    """Test prompt building functionality."""

    def test_build_rag_prompt_with_documents(self):
        """Test building RAG prompt with subtitle documents."""
        query = "What did Tony Stark say about technology?"
        documents = [
            Document(
                content="Technology should be used to help people, not harm them.",
                meta={
                    "start_ms": 15000,
                    "end_ms": 18000,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.95
                }
            ),
            Document(
                content="Innovation is the key to solving our biggest challenges.",
                meta={
                    "start_ms": 25000,
                    "end_ms": 28000,
                    "speaker": "Tony Stark",
                    "retrieval_score": 0.87
                }
            )
        ]
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_max_context_length=4000)
            
            prompt = build_rag_prompt(query, documents)
            
            # Verify prompt structure
            assert "Context from subtitles/transcripts:" in prompt
            assert query in prompt
            assert "Technology should be used to help people" in prompt
            assert "Innovation is the key to solving" in prompt
            assert "Tony Stark" in prompt
            assert "15.0s - 18.0s" in prompt

    def test_build_rag_prompt_without_documents(self):
        """Test building prompt when no documents are provided."""
        query = "What is artificial intelligence?"
        documents = []
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_max_context_length=4000)
            
            prompt = build_rag_prompt(query, documents)
            
            # Should use query-only format
            assert "Please answer the following question:" in prompt
            assert query in prompt
            assert "Context from subtitles" not in prompt

    def test_build_rag_prompt_respects_context_length_limit(self):
        """Test that prompt building respects context length limits."""
        query = "Test query"
        
        # Create many documents to exceed limit
        documents = []
        for i in range(20):
            documents.append(Document(
                content=f"This is a long piece of dialogue from character {i} that talks about various topics. " * 10,
                meta={"speaker": f"Character {i}", "start_ms": i * 1000, "end_ms": (i + 1) * 1000}
            ))
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_max_context_length=500)  # Small limit
            
            prompt = build_rag_prompt(query, documents)
            
            # Should not include all documents due to length limit
            # The prompt should be longer than just the base prompt but not include all docs
            base_prompt_length = len(_build_query_only_prompt(query))
            assert len(prompt) > base_prompt_length  # Has some context
            assert "Character 0" in prompt  # First document should be included
            # With a 500 char limit, we shouldn't get all 20 characters
            character_count = sum(1 for i in range(20) if f"Character {i}" in prompt)
            assert character_count < 20  # Not all documents included due to limit

    def test_format_document_context_with_full_metadata(self):
        """Test formatting document with complete metadata."""
        document = Document(
            content="Hello there, how are you doing today?",
            meta={
                "start_ms": 15500,
                "end_ms": 18750,
                "speaker": "Character A",
                "retrieval_score": 0.923
            },
            score=0.923
        )
        
        formatted = _format_document_context(document, 1)
        
        assert "[1]" in formatted
        assert "15.5s - 18.8s" in formatted
        assert "Character A" in formatted
        assert "0.923" in formatted
        assert "Hello there, how are you doing today?" in formatted

    def test_format_document_context_with_minimal_metadata(self):
        """Test formatting document with minimal metadata."""
        document = Document(
            content="Simple dialogue without much metadata.",
            meta={}
        )
        
        formatted = _format_document_context(document, 2)
        
        assert "[2]" in formatted
        assert "Simple dialogue without much metadata." in formatted
        # Should not crash with missing metadata

    def test_build_system_prompt_general(self):
        """Test building general system prompt."""
        prompt = build_system_prompt("general")
        
        assert "helpful assistant" in prompt.lower()
        assert "subtitle" in prompt.lower() or "transcript" in prompt.lower()

    def test_build_system_prompt_character_analysis(self):
        """Test building character analysis system prompt."""
        prompt = build_system_prompt("character_analysis")
        
        assert "character" in prompt.lower()
        assert "analysis" in prompt.lower()

    def test_build_system_prompt_quote_finding(self):
        """Test building quote finding system prompt."""
        prompt = build_system_prompt("quote_finding")
        
        assert "quote" in prompt.lower()
        assert "dialogue" in prompt.lower() or "lines" in prompt.lower()

    def test_build_system_prompt_timeline(self):
        """Test building timeline system prompt."""
        prompt = build_system_prompt("timeline")
        
        assert "timeline" in prompt.lower()
        assert "when" in prompt.lower() or "sequence" in prompt.lower()

    def test_build_system_prompt_unknown_type(self):
        """Test building system prompt with unknown type defaults to general."""
        prompt = build_system_prompt("unknown_type")
        
        # Should return general prompt
        general_prompt = build_system_prompt("general")
        assert prompt == general_prompt

    def test_truncate_context_to_limit_no_truncation_needed(self):
        """Test truncation when context is within limit."""
        context = "Short context that fits within limits."
        max_length = 1000
        
        result = truncate_context_to_limit(context, max_length)
        
        assert result == context

    def test_truncate_context_to_limit_truncation_needed(self):
        """Test truncation when context exceeds limit."""
        context = "Very long context " * 100  # Make it long
        max_length = 200
        
        result = truncate_context_to_limit(context, max_length)
        
        assert len(result) <= max_length
        assert "truncated due to length limits" in result

    def test_truncate_context_smart_boundary_breaking(self):
        """Test that truncation tries to break at document boundaries."""
        context = "[1] First document content\n[2] Second document content\n[3] Third document content"
        max_length = 60  # Should fit first document but not much more
        
        result = truncate_context_to_limit(context, max_length)
        
        # Should break at a document boundary if possible
        assert "[1]" in result
        assert "truncated" in result

    def test_build_query_only_prompt(self):
        """Test building prompt for query without context."""
        query = "What is the meaning of life?"
        
        prompt = _build_query_only_prompt(query)
        
        assert "Please answer the following question:" in prompt
        assert query in prompt
        assert "Answer:" in prompt

    def test_build_contextualized_prompt(self):
        """Test building prompt with context."""
        query = "Who said this quote?"
        context = "[1] (Time: 10.0s - 12.0s | Speaker: Character A)\nHello world!"
        
        prompt = _build_contextualized_prompt(query, context)
        
        assert "helpful assistant" in prompt
        assert "Context from subtitles/transcripts:" in prompt
        assert context in prompt
        assert query in prompt
        assert "Answer:" in prompt

    def test_build_rag_prompt_preserves_document_order(self):
        """Test that document order is preserved in the prompt."""
        query = "Test query"
        documents = [
            Document(content="First document", meta={"start_ms": 1000}),
            Document(content="Second document", meta={"start_ms": 2000}),
            Document(content="Third document", meta={"start_ms": 3000}),
        ]
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_max_context_length=4000)
            
            prompt = build_rag_prompt(query, documents)
            
            # Check that documents appear in order
            first_pos = prompt.find("First document")
            second_pos = prompt.find("Second document")
            third_pos = prompt.find("Third document")
            
            assert first_pos < second_pos < third_pos