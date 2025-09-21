"""End-to-end Haystack pipeline: SRT → preprocess → chunk → embed → Qdrant.

This module provides:
- ChunkJsonlToTexts: Convert chunk JSONL lines to aligned lists of texts and metadata.
- BuildDocuments: Combine texts, metas, and embeddings into Qdrant-ready document dicts.
- build_srt_to_qdrant_pipeline(): Returns a Haystack Pipeline wired with existing components
  (SRTPreprocessor, SemanticChunker, TextEmbedder, QdrantWriter) and these glue components.

Usage example:

from haystack.core.pipeline import Pipeline
from src.pipelines.srt_to_qdrant import build_srt_to_qdrant_pipeline

pipe = build_srt_to_qdrant_pipeline()
result = pipe.run({"pre": {"srt_text": "... raw .srt content ..."}})
written = result["write"]["stats"]["written"]

"""

import json
from typing import Any, Dict, List

from haystack.core.pipeline import Pipeline
from haystack.core.component import component
from src.core.logging_config import get_logger, LoggedOperation, MetricsLogger


@component
class ChunkJsonlToTexts:
    """Parse chunk JSONL lines into texts and metadata.

    Inputs:
        - chunk_lines: list[str] — JSONL lines with at least a "text" field; optional
          fields like start_ms, end_ms, caption_indices, token_count are preserved in meta.

    Outputs:
        - texts: list[str]
        - metas: list[dict]
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        self._logger.info(
            "ChunkJsonlToTexts initialized",
            component="pipeline_glue"
        )

    @component.output_types(texts=List[str], metas=List[Dict[str, Any]])
    def run(self, chunk_lines: List[str]) -> dict[str, object]:  # type: ignore[override]
        with LoggedOperation("chunk_jsonl_parsing", self._logger, input_lines=len(chunk_lines)) as op:
            texts: list = []
            metas: List[Dict[str, Any]] = []
            skipped_lines = 0
            
            for i, line in enumerate(chunk_lines):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines instead of failing the whole pipeline
                    self._logger.warning(
                        "Invalid JSON line skipped",
                        line_index=i,
                        line_content=line[:100] + "..." if len(line) > 100 else line,
                        component="pipeline_glue"
                    )
                    skipped_lines += 1
                    continue
                    
                text = obj.get("text")
                if not isinstance(text, str):
                    self._logger.warning(
                        "Line missing valid text field",
                        line_index=i,
                        text_type=type(text).__name__,
                        component="pipeline_glue"
                    )
                    skipped_lines += 1
                    continue
                    
                meta = {k: v for k, v in obj.items() if k != "text"}
                texts.append(text)
                metas.append(meta)
                
                self._logger.debug(
                    "Chunk line processed",
                    line_index=i,
                    text_length=len(text),
                    meta_fields=list(meta.keys()),
                    component="pipeline_glue"
                )
            
            # Add context and metrics
            op.add_context(
                output_texts=len(texts),
                skipped_lines=skipped_lines,
                success_rate=len(texts) / len(chunk_lines) if chunk_lines else 0
            )
            
            self._metrics.counter("chunk_lines_processed_total", len(chunk_lines), component="pipeline_glue")
            self._metrics.counter("texts_extracted_total", len(texts), component="pipeline_glue", status="success")
            if skipped_lines > 0:
                self._metrics.counter("chunk_lines_skipped_total", skipped_lines, component="pipeline_glue", reason="invalid")
            
            self._logger.info(
                "Chunk JSONL parsing completed",
                input_lines=len(chunk_lines),
                output_texts=len(texts),
                skipped_lines=skipped_lines,
                component="pipeline_glue"
            )
            
            return {"texts": texts, "metas": metas}


@component
class BuildDocuments:
    """Build QdrantWriter-compatible documents from texts, metas, and embeddings.

    Inputs:
        - texts: list[str]
        - metas: list[dict]
        - embeddings: list[list[float]]

    Output:
        - documents: list[dict] where each item has keys: content, embedding, meta
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._metrics = MetricsLogger(self._logger)
        
        self._logger.info(
            "BuildDocuments initialized",
            component="pipeline_glue"
        )

    @component.output_types(documents=List[Dict[str, Any]])
    def run(
        self,
        texts: List[str],
        metas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> dict[str, object]:  # type: ignore[override]
        with LoggedOperation("document_building", self._logger, 
                           input_texts=len(texts), 
                           input_metas=len(metas), 
                           input_embeddings=len(embeddings)) as op:
            
            # Check input alignment
            input_lengths = [len(texts), len(metas), len(embeddings)]
            n = min(input_lengths)
            max_length = max(input_lengths)
            
            if max_length != n:
                self._logger.warning(
                    "Input arrays have mismatched lengths, using minimum",
                    text_count=len(texts),
                    meta_count=len(metas),
                    embedding_count=len(embeddings),
                    using_count=n,
                    component="pipeline_glue"
                )
            
            docs: List[Dict[str, Any]] = []
            for i in range(n):
                try:
                    doc = {
                        "content": texts[i],
                        "embedding": embeddings[i],
                        "meta": metas[i] if metas[i] is not None else {},
                    }
                    docs.append(doc)
                    
                    self._logger.debug(
                        "Document built successfully",
                        document_index=i,
                        content_length=len(texts[i]),
                        embedding_dimension=len(embeddings[i]),
                        meta_fields=list(metas[i].keys()) if metas[i] else [],
                        component="pipeline_glue"
                    )
                    
                except Exception as e:
                    self._logger.error(
                        "Failed to build document",
                        document_index=i,
                        error=str(e),
                        error_type=type(e).__name__,
                        component="pipeline_glue"
                    )
                    continue
            
            # Add context and metrics
            op.add_context(
                output_documents=len(docs),
                input_alignment_ratio=n / max_length if max_length > 0 else 0,
                documents_built=len(docs)
            )
            
            self._metrics.counter("documents_built_total", len(docs), component="pipeline_glue", status="success")
            self._metrics.histogram("document_build_batch_size", len(docs), component="pipeline_glue")
            
            if max_length != n:
                self._metrics.counter("input_misalignment_total", 1, component="pipeline_glue", 
                                    texts=len(texts), metas=len(metas), embeddings=len(embeddings))
            
            self._logger.info(
                "Document building completed",
                input_texts=len(texts),
                input_metas=len(metas),
                input_embeddings=len(embeddings),
                output_documents=len(docs),
                component="pipeline_glue"
            )
            
            return {"documents": docs}


def build_srt_to_qdrant_pipeline():
    """Construct the end-to-end pipeline.

    Graph:
        pre(SRTPreprocessor).jsonl_lines → chunk(SemanticChunker).jsonl_lines
        chunk.chunk_lines → explode(ChunkJsonlToTexts).chunk_lines
        explode.texts → embed(TextEmbedder).text    (mapped across list)
        explode.texts, explode.metas, embed.embedding → docs(BuildDocuments)
        docs.documents → write(QdrantWriter).documents
    """
    logger = get_logger(__name__)
    
    logger.info(
        "Building SRT-to-Qdrant pipeline",
        pipeline_type="srt_to_qdrant",
        component="pipeline_builder"
    )
    
    # Import components locally to avoid import-time side effects when this module is imported for type checking
    from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
    from src.components.chunker.semantic_chunker import SemanticChunker
    from src.components.embedder.text_embedder import TextEmbedder
    from src.components.writer.qdrant_writer import QdrantWriter

    try:
        pipe = Pipeline()
        
        # Add components with logging
        logger.debug("Adding SRTPreprocessor component", component="pipeline_builder")
        pipe.add_component("pre", SRTPreprocessor())
        
        logger.debug("Adding SemanticChunker component", component="pipeline_builder")
        pipe.add_component("chunk", SemanticChunker())
        pipe.connect("pre.jsonl_lines", "chunk.jsonl_lines")

        logger.debug("Adding ChunkJsonlToTexts component", component="pipeline_builder")
        pipe.add_component("explode", ChunkJsonlToTexts())
        pipe.connect("chunk.chunk_lines", "explode.chunk_lines")

        logger.debug("Adding TextEmbedder component", component="pipeline_builder")
        pipe.add_component("embed", TextEmbedder())
        pipe.connect("explode.texts", "embed.text")  # List[str] → List[str]

        logger.debug("Adding BuildDocuments component", component="pipeline_builder")
        pipe.add_component("docs", BuildDocuments())
        pipe.connect("explode.texts", "docs.texts")
        pipe.connect("explode.metas", "docs.metas")
        pipe.connect("embed.embedding", "docs.embeddings")

        logger.debug("Adding QdrantWriter component", component="pipeline_builder")
        pipe.add_component("write", QdrantWriter())
        pipe.connect("docs.documents", "write.documents")

        # Log pipeline graph information
        component_names = list(pipe.graph.nodes())
        connections = list(pipe.graph.edges())
        
        logger.info(
            "SRT-to-Qdrant pipeline built successfully",
            total_components=len(component_names),
            component_names=component_names,
            total_connections=len(connections),
            pipeline_type="srt_to_qdrant",
            component="pipeline_builder"
        )
        
        return pipe
        
    except Exception as e:
        logger.error(
            "Failed to build SRT-to-Qdrant pipeline",
            error=str(e),
            error_type=type(e).__name__,
            component="pipeline_builder"
        )
        raise
