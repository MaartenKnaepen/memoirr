"""End-to-end Haystack pipeline: SRT → preprocess → chunk → embed → Qdrant.

This module provides:
- ChunkJsonlToTexts: Convert chunk JSONL lines to aligned lists of texts and metadata.
- BuildDocuments: Combine texts, metas, and embeddings into Qdrant-ready document dicts.
- build_srt_to_qdrant_pipeline(): Returns a Haystack Pipeline wired with existing components
  (SRTPreprocessor, SemanticChunker, TextEmbedder, QdrantWriter) and these glue components.

Usage example:

from haystack import Pipeline
from src.pipelines.srt_to_qdrant import build_srt_to_qdrant_pipeline

pipe = build_srt_to_qdrant_pipeline()
result = pipe.run({"pre": {"srt_text": "... raw .srt content ..."}})
written = result["write"]["stats"]["written"]

"""

import json
from typing import Any, Dict, List

from haystack import Pipeline, component


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

    @component.output_types(texts=List[str], metas=List[Dict[str, Any]])
    def run(self, chunk_lines: List[str]) -> dict[str, object]:  # type: ignore[override]
        texts: list = []
        metas: List[Dict[str, Any]] = []
        for line in chunk_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines instead of failing the whole pipeline
                continue
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            meta = {k: v for k, v in obj.items() if k != "text"}
            texts.append(text)
            metas.append(meta)
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

    @component.output_types(documents=List[Dict[str, Any]])
    def run(
        self,
        texts: List[str],
        metas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> dict[str, object]:  # type: ignore[override]
        n = min(len(texts), len(metas), len(embeddings))
        docs: List[Dict[str, Any]] = []
        for i in range(n):
            docs.append({
                "content": texts[i],
                "embedding": embeddings[i],
                "meta": metas[i] if metas[i] is not None else {},
            })
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
    # Import components locally to avoid import-time side effects when this module is imported for type checking
    from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
    from src.components.chunker.semantic_chunker import SemanticChunker
    from src.components.embedder.text_embedder import TextEmbedder
    from src.components.writer.qdrant_writer import QdrantWriter

    pipe = Pipeline()
    pipe.add_component("pre", SRTPreprocessor())
    pipe.add_component("chunk", SemanticChunker())
    pipe.connect("pre.jsonl_lines", "chunk.jsonl_lines")

    pipe.add_component("explode", ChunkJsonlToTexts())
    pipe.connect("chunk.chunk_lines", "explode.chunk_lines")

    pipe.add_component("embed", TextEmbedder())
    pipe.connect("explode.texts", "embed.text")  # List[str] → List[str]

    pipe.add_component("docs", BuildDocuments())
    pipe.connect("explode.texts", "docs.texts")
    pipe.connect("explode.metas", "docs.metas")
    pipe.connect("embed.embedding", "docs.embeddings")

    pipe.add_component("write", QdrantWriter())
    pipe.connect("docs.documents", "write.documents")

    return pipe
