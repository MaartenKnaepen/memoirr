import pytest
from haystack import Pipeline

from src.components.preprocessor.srt_preprocessor import SRTPreprocessor
from src.components.chunker.semantic_chunker import SemanticChunker


def test_pipeline_connect_explicit_output_to_input():
    """Expect to connect preprocessor jsonl_lines to chunker jsonl_lines without errors.

    This test will fail until the component socket types are aligned for the current Haystack version.
    """
    pipe = Pipeline()
    pipe.add_component("pre", SRTPreprocessor())
    pipe.add_component("chunk", SemanticChunker())

    # Explicit sockets
    pipe.connect("pre.jsonl_lines", "chunk.jsonl_lines")


def test_pipeline_connect_shorthand_receiver():
    """Expect to connect using explicit sender socket and receiver component name.

    Receiver shorthand is valid only if the component has a single input (chunker does).
    This test will fail until the component socket types are aligned.
    """
    pipe = Pipeline()
    pipe.add_component("pre", SRTPreprocessor())
    pipe.add_component("chunk", SemanticChunker())

    # Sender must be explicit since pre has multiple outputs; receiver can be shorthand
    pipe.connect("pre.jsonl_lines", "chunk")
