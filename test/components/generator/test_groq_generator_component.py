"""Tests for GroqGenerator Haystack component.

Tests the component interface, configuration, and integration with mocked dependencies.
Follows Memoirr testing patterns: comprehensive mocking, error scenarios, metrics validation.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document
from src.components.generator.groq_generator import GroqGenerator


class TestGroqGeneratorComponent:
    """Test the GroqGenerator Haystack component."""

    def test_groq_generator_initializes_with_defaults(self):
        """Test that component initializes with default configuration."""
        from src.core.config import get_settings
        
        # Get actual settings from config/env
        settings = get_settings()
        
        generator = GroqGenerator()
        
        # Test that component uses the actual configured values
        assert generator.model == settings.groq_model
        assert generator.system_prompt_template == settings.groq_system_prompt_template
        assert generator.max_tokens == settings.groq_max_tokens
        assert generator.temperature == settings.groq_temperature
        assert generator.top_p == settings.groq_top_p
        assert generator.stream == settings.groq_stream

    def test_groq_generator_initializes_with_custom_params(self):
        """Test that component respects custom initialization parameters."""
        custom_template = "custom_system.j2"
        generator = GroqGenerator(
            model="mixtral-8x7b-32768",
            system_prompt_template=custom_template,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            stream=True,
        )
        
        # Test that custom parameters override config values
        assert generator.model == "mixtral-8x7b-32768"
        assert generator.system_prompt_template == custom_template
        assert generator.max_tokens == 2048
        assert generator.temperature == 0.3
        assert generator.top_p == 0.9
        assert generator.stream is True

    def test_groq_generator_run_returns_replies_and_meta(self):
        """Test that run method generates and returns replies with metadata."""
        from src.core.config import get_settings
        
        # Get actual settings to use in mock data
        settings = get_settings()
        
        mock_replies = ["This is a generated response about the movie scene."]
        mock_meta = [{
            "model": settings.groq_model,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200
            },
            "query": "test query",
            "document_count": 1
        }]
        
        with patch('src.components.generator.groq_generator.orchestrate_generation') as mock_orchestrate:
            mock_orchestrate.return_value = (mock_replies, mock_meta)
            
            generator = GroqGenerator()
            test_documents = [Document(content="Test movie dialogue", meta={"speaker": "Character A"})]
            
            result = generator.run(
                query="What did Character A say?",
                documents=test_documents
            )
            
            assert "replies" in result
            assert "meta" in result
            assert len(result["replies"]) == 1
            assert result["replies"][0] == "This is a generated response about the movie scene."
            assert len(result["meta"]) == 1
            assert result["meta"][0]["model"] == settings.groq_model
            
            # Verify orchestrate_generation was called with correct parameters
            mock_orchestrate.assert_called_once()
            call_args = mock_orchestrate.call_args[1]
            assert call_args["query"] == "What did Character A say?"
            assert len(call_args["documents"]) == 1
            assert call_args["model"] == settings.groq_model

    def test_groq_generator_run_with_parameter_overrides(self):
        """Test that run method respects parameter overrides."""
        from src.core.config import get_settings
        
        # Get actual settings to use in mock data
        settings = get_settings()
        
        mock_replies = ["Override test response."]
        mock_meta = [{"model": settings.groq_model, "finish_reason": "stop"}]
        
        with patch('src.components.generator.groq_generator.orchestrate_generation') as mock_orchestrate:
            mock_orchestrate.return_value = (mock_replies, mock_meta)
            
            generator = GroqGenerator()
            
            # Override parameters in run call
            result = generator.run(
                query="Test query with overrides",
                documents=[],
                task_type="character_analysis",
                custom_instructions="Focus on character relationships",
                max_tokens=512,
                temperature=0.2,
                top_p=0.8
            )
            
            assert "replies" in result
            assert result["replies"] == mock_replies
            
            # Verify overrides were passed to orchestrator
            call_args = mock_orchestrate.call_args[1]
            # Should contain rendered template with task type and custom instructions
            assert "character_analysis" in call_args["system_prompt"]
            assert "Focus on character relationships" in call_args["system_prompt"]
            assert call_args["max_tokens"] == 512
            assert call_args["temperature"] == 0.2
            assert call_args["top_p"] == 0.8

    def test_groq_generator_run_without_documents(self):
        """Test that run method handles queries without retrieved documents."""
        from src.core.config import get_settings
        
        # Get actual settings to use in mock data
        settings = get_settings()
        
        mock_replies = ["I can answer general questions without specific context."]
        mock_meta = [{"model": settings.groq_model, "document_count": 0}]
        
        with patch('src.components.generator.groq_generator.orchestrate_generation') as mock_orchestrate:
            mock_orchestrate.return_value = (mock_replies, mock_meta)
            
            generator = GroqGenerator()
            
            # No documents provided
            result = generator.run(query="General question without context")
            
            assert "replies" in result
            assert result["replies"] == mock_replies
            
            # Verify empty documents list was passed
            call_args = mock_orchestrate.call_args[1]
            assert call_args["documents"] == []

    def test_groq_generator_run_handles_orchestrator_errors(self):
        """Test that run method properly handles and propagates orchestrator errors."""
        with patch('src.components.generator.groq_generator.orchestrate_generation') as mock_orchestrate:
            mock_orchestrate.side_effect = RuntimeError("Groq API error")
            
            generator = GroqGenerator()
            
            with pytest.raises(RuntimeError, match="Groq API error"):
                generator.run(query="error query")

    def test_groq_generator_component_output_types(self):
        """Test that component declares correct output types for Haystack compatibility."""
        generator = GroqGenerator()
        
        # Check that the component has the expected output types
        assert hasattr(generator, '__haystack_output__')
        output_sockets = getattr(generator, '__haystack_output__')
        assert hasattr(output_sockets, '_sockets_dict')
        sockets_dict = output_sockets._sockets_dict
        assert 'replies' in sockets_dict
        assert 'meta' in sockets_dict

    def test_groq_generator_logs_metrics_correctly(self):
        """Test that the component logs appropriate metrics during generation."""
        from src.core.config import get_settings
        
        # Get actual settings to use in mock data
        settings = get_settings()
        
        mock_replies = ["Test reply"]
        mock_meta = [{
            "model": settings.groq_model,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125
            }
        }]
        
        with patch('src.components.generator.groq_generator.orchestrate_generation') as mock_orchestrate:
            mock_orchestrate.return_value = (mock_replies, mock_meta)
            
            generator = GroqGenerator()
            
            # Mock the metrics logger to capture calls
            with patch.object(generator._metrics, 'counter') as mock_counter:
                with patch.object(generator._metrics, 'histogram') as mock_histogram:
                    result = generator.run(query="test query")
                    
                    # Verify metrics were recorded
                    assert mock_counter.called
                    assert mock_histogram.called
                    
                    # Check specific metric calls
                    counter_calls = [call[0][0] for call in mock_counter.call_args_list]
                    assert "generation_requests_total" in counter_calls
                    assert "replies_generated_total" in counter_calls