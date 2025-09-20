"""Tests for response processing utilities.

Tests response parsing, metadata extraction, and formatting from Groq API responses.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from haystack.dataclasses import Document
from src.components.generator.utilities.groq_generator.response_processor import (
    process_groq_response,
    format_reply_for_display,
    _extract_reply_text,
    _build_reply_metadata,
    _extract_context_sources,
    _extract_time_range,
    _extract_speakers,
)


class TestResponseProcessor:
    """Test response processing functionality."""

    def create_mock_groq_response(self, choices=None, usage=None, response_id="test_123"):
        """Helper to create mock Groq response objects."""
        if choices is None:
            choices = [
                MagicMock(
                    message=MagicMock(content="This is a test response."),
                    finish_reason="stop",
                    index=0
                )
            ]
        
        if usage is None:
            usage = MagicMock(
                prompt_tokens=150,
                completion_tokens=25,
                total_tokens=175
            )
        
        response = MagicMock()
        response.choices = choices
        response.usage = usage
        response.id = response_id
        response.created = 1234567890
        response.object = "chat.completion"
        
        return response

    def test_process_groq_response_single_choice(self):
        """Test processing response with single choice."""
        response = self.create_mock_groq_response()
        model = "llama3-8b-8192"
        query = "What did the character say?"
        documents = [
            Document(
                content="Hello there!",
                meta={"speaker": "Character A", "start_ms": 5000, "end_ms": 7000}
            )
        ]
        generation_params = {"temperature": 0.7, "max_tokens": 1024}
        
        replies, meta = process_groq_response(
            response=response,
            model=model,
            query=query,
            documents=documents,
            generation_params=generation_params
        )
        
        assert len(replies) == 1
        assert replies[0] == "This is a test response."
        
        assert len(meta) == 1
        assert meta[0]["model"] == model
        assert meta[0]["finish_reason"] == "stop"
        assert meta[0]["query"] == query
        assert meta[0]["document_count"] == 1
        assert meta[0]["usage"]["total_tokens"] == 175

    def test_process_groq_response_multiple_choices(self):
        """Test processing response with multiple choices."""
        choices = [
            MagicMock(
                message=MagicMock(content="First response option."),
                finish_reason="stop",
                index=0
            ),
            MagicMock(
                message=MagicMock(content="Second response option."),
                finish_reason="stop", 
                index=1
            )
        ]
        
        response = self.create_mock_groq_response(choices=choices)
        
        replies, meta = process_groq_response(
            response=response,
            model="llama3-8b-8192",
            query="test query",
            documents=[],
            generation_params={}
        )
        
        assert len(replies) == 2
        assert replies[0] == "First response option."
        assert replies[1] == "Second response option."
        
        assert len(meta) == 2
        assert meta[0]["choice_index"] == 0
        assert meta[1]["choice_index"] == 1

    def test_process_groq_response_no_choices(self):
        """Test handling response with no choices."""
        response = self.create_mock_groq_response(choices=[])
        
        replies, meta = process_groq_response(
            response=response,
            model="llama3-8b-8192",
            query="test query",
            documents=[],
            generation_params={}
        )
        
        assert replies == []
        assert meta == []

    def test_process_groq_response_invalid_choice(self):
        """Test handling response with invalid choice structure."""
        # Choice with missing message content
        bad_choice = MagicMock()
        bad_choice.message = MagicMock()
        bad_choice.message.content = None
        bad_choice.finish_reason = "stop"
        
        good_choice = MagicMock(
            message=MagicMock(content="Good response."),
            finish_reason="stop",
            index=1
        )
        
        response = self.create_mock_groq_response(choices=[bad_choice, good_choice])
        
        replies, meta = process_groq_response(
            response=response,
            model="llama3-8b-8192",
            query="test query",
            documents=[],
            generation_params={}
        )
        
        # Should skip bad choice and process good one
        assert len(replies) == 1
        assert replies[0] == "Good response."
        assert len(meta) == 1

    def test_extract_reply_text_valid_choice(self):
        """Test extracting text from valid choice."""
        choice = MagicMock()
        choice.message = MagicMock()
        choice.message.content = "  Test response with whitespace.  "
        
        text = _extract_reply_text(choice)
        
        assert text == "Test response with whitespace."

    def test_extract_reply_text_invalid_choice(self):
        """Test extracting text from invalid choice raises error."""
        choice = MagicMock()
        choice.message = MagicMock()
        choice.message.content = None
        
        with pytest.raises(ValueError, match="Invalid choice format"):
            _extract_reply_text(choice)

    def test_build_reply_metadata_complete(self):
        """Test building metadata with all available information."""
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.index = 0
        
        response = MagicMock()
        response.id = "response_123"
        response.created = 1234567890
        response.object = "chat.completion"
        response.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        documents = [
            Document(
                content="Test content",
                meta={"speaker": "Test Speaker", "start_ms": 1000}
            )
        ]
        
        meta = _build_reply_metadata(
            choice=choice,
            choice_index=0,
            response=response,
            model="llama3-8b-8192",
            query="test query",
            documents=documents,
            generation_params={"temperature": 0.7}
        )
        
        assert meta["model"] == "llama3-8b-8192"
        assert meta["finish_reason"] == "stop"
        assert meta["query"] == "test query"
        assert meta["document_count"] == 1
        assert meta["usage"]["total_tokens"] == 150
        assert meta["response_id"] == "response_123"
        assert len(meta["context_sources"]) == 1

    def test_extract_context_sources(self):
        """Test extracting source information from documents."""
        documents = [
            Document(
                content="First document with some content here.",
                meta={
                    "speaker": "Character A",
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "retrieval_score": 0.95
                },
                score=0.95
            ),
            Document(
                content="Short doc",
                meta={"speaker": "Character B"},
                score=0.87
            )
        ]
        
        sources = _extract_context_sources(documents)
        
        assert len(sources) == 2
        
        # First document
        assert sources[0]["index"] == 0
        assert sources[0]["speaker"] == "Character A"
        assert sources[0]["start_ms"] == 1000
        assert sources[0]["end_ms"] == 3000
        assert sources[0]["retrieval_score"] == 0.95
        assert sources[0]["similarity_score"] == 0.95
        assert "First document with some" in sources[0]["content_preview"]
        
        # Second document
        assert sources[1]["index"] == 1
        assert sources[1]["speaker"] == "Character B"
        assert sources[1]["content_preview"] == "Short doc"
        assert sources[1]["similarity_score"] == 0.87

    def test_extract_time_range(self):
        """Test extracting time range from sources."""
        sources = [
            {"start_ms": 5000, "end_ms": 8000},
            {"start_ms": 12000, "end_ms": 15000},
            {"start_ms": 20000, "end_ms": 25000},
        ]
        
        time_range = _extract_time_range(sources)
        
        assert time_range is not None
        assert time_range["start_seconds"] == 5.0
        assert time_range["end_seconds"] == 25.0
        assert time_range["duration_seconds"] == 20.0

    def test_extract_time_range_no_timing(self):
        """Test extracting time range when no timing info available."""
        sources = [
            {"speaker": "Character A"},
            {"content": "Some content"},
        ]
        
        time_range = _extract_time_range(sources)
        
        assert time_range is None

    def test_extract_speakers(self):
        """Test extracting unique speakers from sources."""
        sources = [
            {"speaker": "Character A"},
            {"speaker": "Character B"},
            {"speaker": "Character A"},  # Duplicate
            {"speaker": "Character C"},
            {},  # No speaker
        ]
        
        speakers = _extract_speakers(sources)
        
        assert len(speakers) == 3
        assert "Character A" in speakers
        assert "Character B" in speakers
        assert "Character C" in speakers
        assert speakers == sorted(speakers)  # Should be sorted

    def test_format_reply_for_display(self):
        """Test formatting reply for user display."""
        reply = "This is the generated answer to your question."
        meta = {
            "model": "llama3-8b-8192",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 25,
                "total_tokens": 175
            },
            "context_sources": [
                {
                    "speaker": "Character A",
                    "start_ms": 5000,
                    "end_ms": 8000
                },
                {
                    "speaker": "Character B", 
                    "start_ms": 12000,
                    "end_ms": 15000
                }
            ]
        }
        
        display = format_reply_for_display(reply, meta)
        
        assert display["answer"] == reply
        assert display["model"] == "llama3-8b-8192"
        assert display["sources_used"] == 2
        assert display["token_usage"]["total"] == 175
        assert display["completion_status"] == "stop"
        
        # Check context summary
        context = display["context_summary"]
        assert context["document_count"] == 2
        assert context["time_range"]["start_seconds"] == 5.0
        assert context["time_range"]["end_seconds"] == 15.0
        assert "Character A" in context["speakers"]
        assert "Character B" in context["speakers"]

    def test_format_reply_for_display_length_warning(self):
        """Test that length truncation warning is added."""
        reply = "Truncated response"
        meta = {
            "model": "llama3-8b-8192",
            "finish_reason": "length",
            "context_sources": []
        }
        
        display = format_reply_for_display(reply, meta)
        
        assert display["completion_status"] == "length"
        assert "warning" in display
        assert "truncated" in display["warning"].lower()

    def test_process_groq_response_handles_none_choices_gracefully(self):
        """Test that None choices are handled gracefully without raising errors."""
        response = MagicMock()
        response.choices = None
        response.id = "test_123"
        
        # Should return empty results instead of raising
        replies, meta = process_groq_response(
            response=response,
            model="llama3-8b-8192",
            query="test query",
            documents=[],
            generation_params={}
        )
        
        assert replies == []
        assert meta == []