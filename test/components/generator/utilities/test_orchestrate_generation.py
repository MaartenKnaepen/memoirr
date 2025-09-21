"""Tests for orchestrate_generation function.

Tests the core generation orchestration logic with comprehensive mocking
of Groq API and utility dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document
from src.components.generator.utilities.groq_generator.orchestrate_generation import orchestrate_generation


class TestOrchestrateGeneration:
    """Test the orchestrate_generation function."""

    def create_mock_groq_response(self, content="Test generated response"):
        """Helper to create mock Groq API response."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(content=content),
                finish_reason="stop",
                index=0
            )
        ]
        response.usage = MagicMock(
            prompt_tokens=150,
            completion_tokens=25,
            total_tokens=175
        )
        response.id = "test_response_123"
        response.created = 1234567890
        response.object = "chat.completion"
        return response

    def test_orchestrate_generation_successful_flow(self):
        """Test successful end-to-end generation orchestration."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        # Get actual settings to use in tests
        from src.core.config import get_settings
        settings = get_settings()
        
        query = "What did the character say about friendship?"
        documents = [
            Document(
                content="Friendship is the most important thing in life.",
                meta={"speaker": "Character A", "start_ms": 10000, "end_ms": 13000}
            )
        ]
        
        mock_response = self.create_mock_groq_response("Friendship is highly valued by the character.")
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_api_key="test_key")
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                mock_build_prompt.return_value = "Mocked RAG prompt with context"
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_groq_class.return_value = mock_client
                    
                    with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.process_groq_response') as mock_process:
                        mock_process.return_value = (
                            ["Friendship is highly valued by the character."],
                            [{"model": settings.groq_model, "usage": {"prompt_tokens": 150, "completion_tokens": 25}}]
                        )
                        
                        # Execute orchestration
                        replies, meta = orchestrate_generation(
                            query=query,
                            documents=documents,
                            model=settings.groq_model,
                            system_prompt="You are a helpful assistant.",
                            max_tokens=1024,
                            temperature=0.7,
                            top_p=1.0,
                            stream=False
                        )
                        
                        # Verify prompt building was called
                        mock_build_prompt.assert_called_once_with(query, documents)
                        
                        # Verify Groq client was initialized (note: api_key comes from local settings call)
                        mock_groq_class.assert_called_once()
                        
                        # Verify API call was made with correct parameters
                        mock_client.chat.completions.create.assert_called_once()
                        call_args = mock_client.chat.completions.create.call_args[1]
                        assert call_args["model"] == settings.groq_model
                        assert call_args["max_tokens"] == 1024
                        assert call_args["temperature"] == 0.7
                        assert call_args["top_p"] == 1.0
                        assert len(call_args["messages"]) == 2  # system + user
                        
                        # Verify response processing was called
                        mock_process.assert_called_once_with(
                            response=mock_response,
                            model=settings.groq_model,
                            query=query,
                            documents=documents,
                            generation_params={
                                "max_tokens": 1024,
                                "temperature": 0.7,
                                "top_p": 1.0
                            }
                        )
                        
                        # Verify results
                        assert len(replies) == 1
                        assert replies[0] == "Friendship is highly valued by the character."
                        assert len(meta) == 1

    def test_orchestrate_generation_without_system_prompt(self):
        """Test generation without system prompt."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        query = "Test query"
        documents = []
        
        mock_response = self.create_mock_groq_response()
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_api_key="test_key")
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                mock_build_prompt.return_value = "Test query prompt"
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_groq_class.return_value = mock_client
                    
                    with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.process_groq_response') as mock_process:
                        mock_process.return_value = (["Response"], [{}])
                        
                        replies, meta = orchestrate_generation(
                            query=query,
                            documents=documents,
                            model=settings.groq_model,
                            system_prompt=None,  # No system prompt
                            max_tokens=512,
                            temperature=0.5,
                            top_p=0.9,
                            stream=False
                        )
                        
                        # Verify only user message was sent (no system message)
                        call_args = mock_client.chat.completions.create.call_args[1]
                        messages = call_args["messages"]
                        assert len(messages) == 1
                        assert messages[0]["role"] == "user"

    def test_orchestrate_generation_validates_input_parameters(self):
        """Test that input parameter validation works correctly."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        documents = []
        
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrate_generation("", documents, settings.groq_model)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrate_generation("   ", documents, settings.groq_model)
        
        # Test invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            orchestrate_generation("query", documents, settings.groq_model, max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            orchestrate_generation("query", documents, settings.groq_model, max_tokens=-1)
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            orchestrate_generation("query", documents, settings.groq_model, temperature=-0.1)
        
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            orchestrate_generation("query", documents, settings.groq_model, temperature=2.1)
        
        # Test invalid top_p
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            orchestrate_generation("query", documents, settings.groq_model, top_p=-0.1)
        
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            orchestrate_generation("query", documents, settings.groq_model, top_p=1.1)

    def test_orchestrate_generation_handles_prompt_building_errors(self):
        """Test error handling when prompt building fails."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
            mock_build_prompt.side_effect = Exception("Prompt building failed")
            
            with pytest.raises(RuntimeError, match="Text generation failed"):
                orchestrate_generation(
                    query="test query",
                    documents=[],
                    model=settings.groq_model
                )

    def test_orchestrate_generation_handles_groq_api_errors(self):
        """Test error handling when Groq API call fails."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_api_key="test_key")
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                mock_build_prompt.return_value = "Test prompt"
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.side_effect = Exception("API call failed")
                    mock_groq_class.return_value = mock_client
                    
                    with pytest.raises(RuntimeError, match="Text generation failed"):
                        orchestrate_generation(
                            query="test query",
                            documents=[],
                            model=settings.groq_model
                        )

    def test_orchestrate_generation_handles_response_processing_errors(self):
        """Test error handling when response processing fails."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        mock_response = self.create_mock_groq_response()
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_api_key="test_key")
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                mock_build_prompt.return_value = "Test prompt"
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_groq_class.return_value = mock_client
                    
                    with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.process_groq_response') as mock_process:
                        mock_process.side_effect = Exception("Response processing failed")
                        
                        with pytest.raises(RuntimeError, match="Text generation failed"):
                            orchestrate_generation(
                                query="test query",
                                documents=[],
                                model=settings.groq_model
                            )

    def test_orchestrate_generation_with_multiple_documents(self):
        """Test generation with multiple context documents."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        query = "What do the characters think about technology?"
        documents = [
            Document(
                content="Technology is amazing and helps everyone.",
                meta={"speaker": "Character A", "start_ms": 5000}
            ),
            Document(
                content="I'm worried about technology taking over.",
                meta={"speaker": "Character B", "start_ms": 15000}
            ),
            Document(
                content="We need to use technology responsibly.",
                meta={"speaker": "Character C", "start_ms": 25000}
            )
        ]
        
        mock_response = self.create_mock_groq_response("Characters have mixed views on technology.")
        
        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock(groq_api_key="test_key")
            
            with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                mock_build_prompt.return_value = "Complex prompt with multiple contexts"
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_groq_class.return_value = mock_client
                    
                    with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.process_groq_response') as mock_process:
                        mock_process.return_value = (
                            ["Characters have mixed views on technology."],
                            [{"document_count": 3}]
                        )
                        
                        replies, meta = orchestrate_generation(
                            query=query,
                            documents=documents,
                            model=settings.groq_model
                        )
                        
                        # Verify prompt was built with all documents
                        mock_build_prompt.assert_called_once_with(query, documents)
                        
                        # Verify response processing received all documents
                        process_call_args = mock_process.call_args[1]
                        assert len(process_call_args["documents"]) == 3

    def test_orchestrate_generation_different_models(self):
        """Test generation with different model names."""
        # Get actual settings to use in test
        from src.core.config import get_settings
        settings = get_settings()
        
        models_to_test = [
            settings.groq_model,
            "mixtral-8x7b-32768", 
            "gemma-7b-it"
        ]
        
        for model in models_to_test:
            mock_response = self.create_mock_groq_response(f"Response from {model}")
            
            with patch('src.core.config.get_settings') as mock_settings:
                mock_settings.return_value = MagicMock(groq_api_key="test_key")
                
                with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.build_rag_prompt') as mock_build_prompt:
                    mock_build_prompt.return_value = "Test prompt"
                    
                    with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.Groq') as mock_groq_class:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create.return_value = mock_response
                        mock_groq_class.return_value = mock_client
                        
                        with patch('src.components.generator.utilities.groq_generator.orchestrate_generation.process_groq_response') as mock_process:
                            mock_process.return_value = ([f"Response from {model}"], [{"model": model}])
                            
                            replies, meta = orchestrate_generation(
                                query="test query",
                                documents=[],
                                model=model
                            )
                            
                            # Verify correct model was used in API call
                            call_args = mock_client.chat.completions.create.call_args[1]
                            assert call_args["model"] == model