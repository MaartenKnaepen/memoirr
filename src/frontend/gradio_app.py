"""Basic Gradio chat interface for the Memoirr RAG pipeline.

This module provides a simple chat interface using Gradio's ChatInterface
component. Users can ask questions about subtitle content and get AI-generated
responses with source attribution.

Usage:
    python -m src.frontend.gradio_app
    
    Or from project root:
    python src/frontend/gradio_app.py
"""

import sys
from pathlib import Path
from typing import List

import gradio as gr

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.rag_pipeline import RAGPipeline
from src.core.logging_config import get_logger


class MemoirrChatInterface:
    """Basic chat interface for the Memoirr RAG pipeline."""
    
    def __init__(self):
        """Initialize the chat interface with RAG pipeline."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Memoirr chat interface")
        
        try:
            self.rag_pipeline = RAGPipeline()
            self.logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def respond(self, message: str, history: List) -> str:
        """Handle chat responses using the RAG pipeline.
        
        Args:
            message: User's question/message
            history: Chat history (handled automatically by Gradio)
            
        Returns:
            AI-generated response with source attribution
        """
        try:
            self.logger.info(f"Processing user query: {message[:100]}...")
            
            # Query the RAG pipeline
            result = self.rag_pipeline.query(message)
            
            # Extract response and sources
            response = result["generator"]["replies"][0]
            documents = result["retriever"]["documents"]
            
            # Format response with sources
            formatted_response = self._format_response_with_sources(response, documents)
            
            self.logger.info("Query processed successfully")
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def _format_response_with_sources(self, response: str, documents: List) -> str:
        """Format the response with source attribution.
        
        Args:
            response: AI-generated response
            documents: Retrieved source documents
            
        Returns:
            Formatted response with sources
        """
        if not documents:
            return f"{response}\n\n*No sources found*"
        
        # Build source list
        sources = []
        for i, doc in enumerate(documents[:3], 1):  # Show top 3 sources
            # Extract metadata
            start_time = doc.meta.get('start_ms', 0) / 1000  # Convert to seconds
            speaker = doc.meta.get('speaker', 'Unknown')
            content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            
            # Format timestamp
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"{minutes}:{seconds:02d}"
            
            sources.append(f"**{i}.** {speaker} at {timestamp} - \"{content_preview}\"")
        
        sources_text = "\n".join(sources)
        
        return f"{response}\n\n### 📽️ Sources:\n{sources_text}"
    
    def create_interface(self) -> gr.ChatInterface:
        """Create the basic Gradio ChatInterface.
        
        Returns:
            Configured Gradio ChatInterface
        """
        self.logger.info("Creating Gradio interface")
        
        interface = gr.ChatInterface(
            fn=self.respond,
            type="messages",
            title="🎬 Memoirr RAG Chat",
            description="Ask questions about subtitle content and get AI-powered answers with source attribution.",
            examples=[
                "What did Tony Stark say about technology?",
                "Who are the main characters in this story?",
                "What happened in the first 10 minutes?",
                "What was the most emotional scene?",
                "Can you summarize the plot?"
            ],
            cache_examples=False,
            theme="soft",
            css="""
                .message-wrap {
                    max-width: 800px;
                }
                .message {
                    font-size: 14px;
                    line-height: 1.5;
                }
            """
        )
        
        self.logger.info("Gradio interface created successfully")
        return interface


def main():
    """Main function to launch the chat interface."""
    try:
        # Create and launch the interface
        chat_app = MemoirrChatInterface()
        demo = chat_app.create_interface()
        
        print("🎬 Starting Memoirr RAG Chat Interface...")
        print("📝 Ask questions about your subtitle content!")
        print("🌐 Interface will be available at: http://localhost:7860")
        
        # Launch the interface
        demo.launch(
            share=False,  # Set to True to create a public link
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()