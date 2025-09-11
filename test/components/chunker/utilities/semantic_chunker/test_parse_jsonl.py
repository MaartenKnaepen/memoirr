import json
import pytest

from src.components.chunker.utilities.semantic_chunker.parse_jsonl import parse_jsonl_lines


def test_parse_jsonl_lines_valid():
    lines = [
        json.dumps({"text": "Hello", "start_ms": 0, "end_ms": 900, "caption_index": 1}),
        json.dumps({"text": "World", "start_ms": 900, "end_ms": 1500, "caption_index": 2}),
    ]
    caps = parse_jsonl_lines(lines)
    assert len(caps) == 2
    assert caps[0].text == "Hello"
    assert caps[1].caption_index == 2


def test_parse_jsonl_lines_fail_fast():
    lines = [
        json.dumps({"text": "OK", "start_ms": 0, "end_ms": 10, "caption_index": 1}),
        "{not-a-json}",
    ]
    with pytest.raises(ValueError):
        parse_jsonl_lines(lines, fail_fast=True)


def test_parse_jsonl_lines_skip_bad():
    lines = [
        "{not-a-json}",
        json.dumps({"text": "OK", "start_ms": 0, "end_ms": 10, "caption_index": 1}),
    ]
    caps = parse_jsonl_lines(lines, fail_fast=False)
    assert len(caps) == 1
    assert caps[0].text == "OK"
