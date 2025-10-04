"""Haystack component to embed text using a local sentence-transformers model.

This component aligns with Memoirr's standards:
- Uses Haystack as the abstraction layer for RAG pipelines.
- Reads configuration from src.core.config.Settings (model name, device, dimension).
- Self-hosted, in-process model loaded from the local models/ folder.

It exposes a single input socket `text` and a single output `embedding`,
following the simple, JSON-serializable socket guidance in documentation/qwen.md.
"""

import os
from typing import List
from contextlib import contextmanager

from haystack.core.component import component
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from tqdm import tqdm

from src.core.config import get_settings
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger
from src.core.model_utils import resolve_model_path
from src.core.memory_utils import clear_gpu_memory, log_memory_usage, check_memory_availability


@contextmanager
def disable_tqdm():
    """Context manager to disable all tqdm progress bars."""
    # Save original environment
    original_disable = os.environ.get('TQDM_DISABLE', None)
    
    # Disable tqdm
    os.environ['TQDM_DISABLE'] = '1'
    
    # Also try to disable sentence-transformers progress bars
    import logging
    sentence_transformers_logger = logging.getLogger('sentence_transformers')
    original_level = sentence_transformers_logger.level
    sentence_transformers_logger.setLevel(logging.WARNING)
    
    try:
        yield
    finally:
        # Restore original environment
        if original_disable is None:
            os.environ.pop('TQDM_DISABLE', None)
        else:
            os.environ['TQDM_DISABLE'] = original_disable
        
        # Restore logging level
        sentence_transformers_logger.setLevel(original_level)


@component
class TextEmbedder:
    """Embed text using a local sentence-transformers model.

    The model folder is resolved relative to the repository's `models/` directory using
    the `EMBEDDING_MODEL_NAME` environment variable (via Settings). This should match
    the model used by the semantic chunker to ensure consistent embedding spaces.
    
    Uses SentenceTransformersDocumentEmbedder for efficient batch processing.
    
    Automatically selects device based on context:
    - CPU for retrieval operations (memory efficient, single queries)
    - GPU for population operations (speed efficient, batch processing)
    """

    def __init__(self, force_device: str = None, context: str = "auto") -> None:
        settings = get_settings()
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        # Smart device selection based on context
        self._device = self._determine_device(force_device, context, settings)
        self._context = context
        
        self._model_name = settings.embedding_model_name
        # Use explicit embedding dimension from settings, with fallback only as last resort
        if settings.embedding_dimension is not None:
            self._embedding_dimension = settings.embedding_dimension
        else:
            # Log warning about using fallback
            self._logger.warning(
                "Embedding dimension not configured, using fallback",
                fallback_dimension=settings.embedding_dimension_fallback,
                component="embedder",
                recommendation="Set EMBEDDING_DIMENSION in .env for proper configuration"
            )
            self._embedding_dimension = settings.embedding_dimension_fallback
        
        # Resolve model directory using common utility
        try:
            model_dir = resolve_model_path(self._model_name)
            self._logger.info(
                "Embedding model resolved successfully",
                model_name=self._model_name,
                model_path=str(model_dir),
                component="embedder"
            )
        except FileNotFoundError as e:
            self._logger.error(
                "Embedding model not found",
                model_name=self._model_name,
                error=str(e),
                component="embedder"
            )
            raise FileNotFoundError(f"Embedding model not found: {e}") from e
        # Initialize embedder with the resolved local path and device
        # Use DocumentEmbedder for efficient batch processing of multiple texts
        # Disable progress bars to avoid individual 1/1 progress bars
        try:
            self._embedder = SentenceTransformersDocumentEmbedder(
                model=str(model_dir),
                device=self._device,
                local_files_only=True,
                show_progress_bar=False,
            )
        except TypeError:
            # Fallback for environments (or tests) where the underlying class signature
            # doesn't accept `local_files_only`, `show_progress_bar`, or `device` (e.g., monkeypatched fakes).
            try:
                self._embedder = SentenceTransformersDocumentEmbedder(
                    model=str(model_dir),
                    device=self._device,
                    show_progress_bar=False,
                )
            except TypeError:
                try:
                    # Try without device parameter
                    self._embedder = SentenceTransformersDocumentEmbedder(
                        model=str(model_dir),
                        show_progress_bar=False,
                    )
                except TypeError:
                    # Final fallback for minimal constructor
                    self._embedder = SentenceTransformersDocumentEmbedder(model=str(model_dir))
        # Warm up so that the first embedding call in a pipeline is not cold.
        self._embedder.warm_up()
        
        self._logger.info(
            "TextEmbedder initialized successfully",
            model_name=self._model_name,
            embedding_dimension=self._embedding_dimension,
            model_path=str(model_dir),
            device=self._device,
            context=self._context,
            component="embedder"
        )

    def _determine_device(self, force_device: str, context: str, settings) -> str:
        """Determine the optimal device based on context and configuration.
        
        Args:
            force_device: Explicit device override (e.g., "cpu", "cuda:0")
            context: Usage context ("retrieval", "population", "auto")
            settings: Application settings
            
        Returns:
            Device string for sentence-transformers
        """
        # 1. Explicit override takes precedence
        if force_device:
            self._logger.info(
                "Using explicitly forced device",
                device=force_device,
                context=context,
                component="embedder"
            )
            return force_device
        
        # 2. Check environment variable override
        if settings.device:
            self._logger.info(
                "Using device from configuration",
                device=settings.device,
                context=context,
                component="embedder"
            )
            return settings.device
            
        # 3. Context-aware automatic selection
        import torch
        
        if context == "retrieval":
            # Always use CPU for retrieval to avoid CUDA memory issues
            device = "cpu"
            self._logger.info(
                "Auto-selected CPU for retrieval context",
                device=device,
                reason="memory_efficiency",
                component="embedder"
            )
        elif context == "population":
            # Use GPU for population if available, else CPU
            cuda_available = torch.cuda.is_available()  # Call ONCE and cache result
            device = "cuda" if cuda_available else "cpu"
            self._logger.info(
                "Auto-selected device for population context",
                device=device,
                cuda_available=cuda_available,  # Use cached result
                reason="batch_processing_speed",
                component="embedder"
            )
        else:
            # Auto context: detect from call stack
            device = self._detect_context_from_stack()
            self._logger.info(
                "Auto-detected device from context",
                device=device,
                component="embedder"
            )
            
        return device
    
    def _detect_context_from_stack(self) -> str:
        """Detect if we're in a retrieval or population context from call stack.
        
        Returns:
            Device string based on detected context
        """
        import inspect
        import torch
        
        # Look through the call stack for context clues
        frame = inspect.currentframe()
        try:
            while frame:
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                
                # Check for retrieval context
                if any(keyword in filename.lower() for keyword in ["retriev", "qdrant_retriever", "orchestrate_retrieval"]):
                    self._context = "retrieval (detected)"
                    return "cpu"  # Always CPU for retrieval
                
                # Check for population context  
                if any(keyword in filename.lower() for keyword in ["population", "batch_processor", "writer", "database"]):
                    self._context = "population (detected)"
                    cuda_available = torch.cuda.is_available()  # Call ONCE and cache result
                    return "cuda" if cuda_available else "cpu"
                
                frame = frame.f_back
        finally:
            del frame
        
        # Default fallback: use CPU to be safe
        self._context = "unknown (safe default)"
        return "cpu"

    @component.output_types(embedding=List[List[float]])
    def run(self, text: List[str]) -> dict[str, object]:  # type: ignore[override]
        """Return embeddings for the provided texts.

        Args:
            text: List of input texts to embed.

        Returns:
            Dict with a single key `embedding` containing the list of list[float] vectors.
        """
        with LoggedOperation("text_embedding", self._logger, input_texts=len(text)) as op:
            # Log memory usage before embedding
            log_memory_usage("before embedding", self._logger)
            
            # Check if we have sufficient memory for batch processing
            memory_available = check_memory_availability()
            if not memory_available:
                self._logger.warning(
                    "Low GPU memory detected, clearing cache before embedding",
                    component="embedder"
                )
                clear_gpu_memory()
            
            try:
                # Convert texts to Documents for batch processing with DocumentEmbedder
                documents = [Document(content=t) for t in text]
                
                # Use batch processing (more efficient)
                with disable_tqdm():
                    result = self._embedder.run(documents)
                
                # Extract embeddings from documents
                embedded_docs = result.get("documents", [])
                if len(embedded_docs) == len(text):
                    embeddings = []
                    for doc in embedded_docs:
                        if hasattr(doc, 'embedding') and doc.embedding is not None:
                            embeddings.append(doc.embedding)
                        else:
                            # Fallback to zero vector if embedding is missing
                            embeddings.append([0.0] * self._embedding_dimension)
                    
                    # Add context and metrics for successful batch processing
                    op.add_context(
                        successful_embeddings=len(embeddings),
                        failed_embeddings=0,
                        processing_mode="batch"
                    )
                    
                    self._metrics.counter("embeddings_generated_total", len(embeddings), component="embedder", mode="batch", status="success")
                    
                    # Log memory usage after successful batch processing
                    log_memory_usage("after batch embedding", self._logger)
                    
                    return {"embedding": embeddings}
            except (TypeError, ValueError, AttributeError) as e:
                # Log the error for debugging but continue with fallback
                self._logger.warning(
                    "Batch processing failed, using individual fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    component="embedder"
                )
                
            # Individual processing fallback with better error handling
            self._logger.info("Using individual embedding processing", text_count=len(text), component="embedder")
            embeddings = []
            failed_count = 0
            
            # Create a single progress bar for all embeddings while disabling individual ones
            with tqdm(total=len(text), desc="Embedding texts", unit="text") as pbar:
                for i, single_text in enumerate(text):
                    try:
                        # Convert single text to Document for DocumentEmbedder
                        single_doc = [Document(content=single_text)]
                        
                        # Disable all tqdm progress bars during individual embedding
                        with disable_tqdm():
                            result = self._embedder.run(single_doc)
                        
                        # Extract embedding from the single document result
                        embedded_docs = result.get("documents", [])
                        if embedded_docs and hasattr(embedded_docs[0], 'embedding') and embedded_docs[0].embedding is not None:
                            embeddings.append(embedded_docs[0].embedding)
                        else:
                            # Fallback to zero vector if embedding is missing
                            embeddings.append([0.0] * self._embedding_dimension)
                            failed_count += 1
                            
                    except Exception as e:
                        # Log the error but continue processing other texts
                        self._logger.warning(
                            "Failed to embed individual text",
                            text_index=i,
                            text_length=len(single_text),
                            error=str(e),
                            error_type=type(e).__name__,
                            fallback_action="using_zero_vector",
                            component="embedder"
                        )
                        # Use zero vector as fallback
                        embeddings.append([0.0] * self._embedding_dimension)
                        failed_count += 1
                    
                    # Update progress bar
                    pbar.update(1)
            
            # Add final context and metrics
            successful_count = len(embeddings) - failed_count
            op.add_context(
                successful_embeddings=successful_count,
                failed_embeddings=failed_count,
                processing_mode="individual"
            )
            
            self._metrics.counter("embeddings_generated_total", successful_count, component="embedder", mode="individual", status="success")
            if failed_count > 0:
                self._metrics.counter("embeddings_failed_total", failed_count, component="embedder", mode="individual", status="failed")
            
            return {"embedding": embeddings}